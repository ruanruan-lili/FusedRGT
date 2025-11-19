import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from rgt_best_model_yuanshi import FusedRGT
import random, numpy as np

# ---------------------------
# 设置随机种子
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# 数据加载与预处理
# ---------------------------
def load_and_preprocess_data(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feats_dict = {
        'bool': torch.load(os.path.join(data_dir, 'node_bool.pt'), map_location=device).float(),
        'cat': torch.load(os.path.join(data_dir, 'node_cat.pt'), map_location=device).float(),
        'num': torch.load(os.path.join(data_dir, 'node_numeric.pt'), map_location=device).float(),
        'desc': torch.load(os.path.join(data_dir, 'node_description.pt'), map_location=device).float(),
        'tweet': torch.load(os.path.join(data_dir, 'node_tweet.pt'), map_location=device).float(),
        'tweet_aug': torch.load(os.path.join(data_dir, 'node_tweet_agu.pt'), map_location=device).float(),
        'net': torch.load(os.path.join(data_dir, 'node_net.pt'), map_location=device).float(),
    }

    # 数值型特征标准化
    numeric_keys = ['bool', 'tweet_aug']
    for key in numeric_keys:
        feat = feats_dict[key]
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True) + 1e-8
        feats_dict[key] = (feat - mean) / std

    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), map_location=device)
    edge_type = torch.load(os.path.join(data_dir, 'edge_type.pt'), map_location=device)
    labels = torch.load(os.path.join(data_dir, 'label.pt'), map_location=device)

    return feats_dict, edge_index, edge_type, labels

# ---------------------------
# 数据加载器
# ---------------------------
def create_data_loaders(feats_dict, edge_index, edge_type, labels, train_mask, val_mask, test_mask,
                        num_neighbors=[15, 10], batch_size=128):
    device = next(iter(feats_dict.values())).device
    num_nodes = labels.size(0)
    x_placeholder = torch.zeros(num_nodes, 1, device=device)

    data = Data(
        x=x_placeholder,
        edge_index=edge_index,
        edge_type=edge_type,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                                  input_nodes=train_mask, shuffle=True)
    val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                                input_nodes=val_mask, shuffle=False)
    test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                                 input_nodes=test_mask, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------
# 训练函数
# ---------------------------
def train_model_with_sampling(model, feats_dict, train_loader, val_loader, test_loader, labels,
                              lr=0.0001, weight_decay=1e-4, epochs=100, patience=20):
    device = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(epochs):
        # ---------------------------
        # 训练阶段
        # ---------------------------
        model.train()
        total_loss = 0
        train_preds, train_labels_list = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            batch_feats = {k: feats_dict[k][batch.n_id].to(device) for k in feats_dict}
            out = model(batch_feats, batch.edge_index, batch.edge_type)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_preds.append(out[batch.train_mask].argmax(dim=1).cpu())
            train_labels_list.append(batch.y[batch.train_mask].cpu())

        train_pred = torch.cat(train_preds, dim=0)
        train_label = torch.cat(train_labels_list, dim=0)
        train_f1 = f1_score(train_label.numpy(), train_pred.numpy(), average='weighted')
        avg_train_loss = total_loss / len(train_loader)

        # ---------------------------
        # 验证阶段
        # ---------------------------
        model.eval()
        val_preds, val_labels_list = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_feats = {k: feats_dict[k][batch.n_id].to(device) for k in feats_dict}
                out = model(batch_feats, batch.edge_index, batch.edge_type)
                loss = criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_loss += loss.item()
                val_preds.append(out[batch.val_mask].argmax(dim=1).cpu())
                val_labels_list.append(batch.y[batch.val_mask].cpu())

        val_pred = torch.cat(val_preds, dim=0)
        val_label = torch.cat(val_labels_list, dim=0)
        val_f1 = f1_score(val_label.numpy(), val_pred.numpy(), average='weighted')
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(val_f1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')

        # 早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_model.pt'))

    # ---------------------------
    # 测试阶段
    # ---------------------------
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_feats = {k: feats_dict[k][batch.n_id].to(device) for k in feats_dict}
            out = model(batch_feats, batch.edge_index, batch.edge_type)
            test_preds.append(out[batch.test_mask].argmax(dim=1).cpu())
            test_labels_list.append(batch.y[batch.test_mask].cpu())

    test_pred = torch.cat(test_preds, dim=0)
    test_label = torch.cat(test_labels_list, dim=0)

    test_acc = accuracy_score(test_label.numpy(), test_pred.numpy())
    test_f1 = f1_score(test_label.numpy(), test_pred.numpy(), average='weighted')
    test_precision = precision_score(test_label.numpy(), test_pred.numpy(), average='weighted', zero_division=0)
    test_recall = recall_score(test_label.numpy(), test_pred.numpy(), average='weighted')

    print("========== Test Results ==========")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print("==================================")

    return model, history

# ---------------------------
# 绘图
# ---------------------------
def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_title('Loss')
    ax2.plot(history['train_f1'], label='Train F1')
    ax2.plot(history['val_f1'], label='Val F1')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1'); ax2.legend(); ax2.set_title('F1 Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ---------------------------
# 主函数
# ---------------------------
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    data_dir = 'feature/sub_feature'
    feats_dict, edge_index, edge_type, labels = load_and_preprocess_data(data_dir)
    labels = labels.long()

    train_mask = torch.load(os.path.join(data_dir, 'train.pt'), map_location=device)
    val_mask = torch.load(os.path.join(data_dir, 'val.pt'), map_location=device)
    test_mask = torch.load(os.path.join(data_dir, 'test.pt'), map_location=device)

    feat_dims = {k: feats_dict[k].shape[1] for k in feats_dict}
    num_classes = len(torch.unique(labels))
    num_relations = len(torch.unique(edge_type))

    # ---------------------------
    # 边丢弃率按类型设置
    # ---------------------------
    edge_dropout_rates = [0.7, 0.7, 0.0, 0.1, 0.3]  # 每种关系的丢弃率
    model = FusedRGT(
        feat_dims,
        hidden_channels=128,
        out_channels=num_classes,
        num_relations=num_relations,
        num_layers=2,
        heads=8,
        dropout=0.5,               # 网络层 dropout
        edge_drop_probs=edge_dropout_rates  # 边丢弃率
    ).to(device)

    train_loader, val_loader, test_loader = create_data_loaders(
        feats_dict, edge_index, edge_type, labels, train_mask, val_mask, test_mask,
        num_neighbors=[15, 10], batch_size=128
    )

    model, history = train_model_with_sampling(
        model, feats_dict, train_loader, val_loader, test_loader, labels,
        lr=0.0001, weight_decay=1e-4, epochs=100, patience=20
    )

    plot_training_history(history, save_path='training_history.png')


if __name__ == "__main__":
    main()
