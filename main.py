# Add this at the top of the file with other imports
from flask import Flask, render_template, jsonify
import threading
import queue
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from models.cgnn import CGNN
from data.data_loader import FraudDataLoader
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import request
import random

app = Flask(__name__)

class CompetitiveGNNModel(nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(CompetitiveGNNModel, self).__init__()
        # GNN layers for processing user behavior graphs
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layer for fraud prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch):
        # Graph convolution operations
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level representations
        x = global_mean_pool(x, batch)
        
        # Final classification
        return self.classifier(x)

class FraudDetectionSystem:
    def __init__(self):
        self.gnn_model = None
        self.baseline_models = {}
        self.metrics = {}

    def construct_user_behavior_graph(self, transactions_df):
        G = nx.Graph()
        for user_id in transactions_df['user_id'].unique():
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            features = {
                'transaction_count': len(user_transactions),
                'avg_amount': user_transactions['amount'].mean(),
                'std_amount': user_transactions['amount'].std(),
                'max_amount': user_transactions['amount'].max()
            }
            G.add_node(user_id, **features)
        
        # Add edges based on transaction patterns
        for _, transaction in transactions_df.iterrows():
            if transaction['receiver_id'] in G.nodes:
                G.add_edge(
                    transaction['user_id'], 
                    transaction['receiver_id'],
                    weight=transaction['amount']
                )
        
        return G

    def train_gnn_model(self, graph_data, labels):
        """
        Train the GNN model for fraud detection
        """
        # Convert networkx graph to PyTorch Geometric data
        data = self._convert_to_pytorch_geometric(graph_data, labels)
        
        # Initialize model
        num_features = data.num_features
        self.gnn_model = CompetitiveGNNModel(num_features)
        
        # Training loop
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        for epoch in range(100):
            self.gnn_model.train()
            optimizer.zero_grad()
            
            out = self.gnn_model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()

    def evaluate_model(self, test_data, test_labels):
        """
        Evaluate model performance and compare with baseline methods
        """
        # GNN model evaluation
        self.gnn_model.eval()
        with torch.no_grad():
            predictions = self.gnn_model(test_data.x, test_data.edge_index, test_data.batch)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, 
            (predictions > 0.5).float(),
            average='binary'
        )
        auc_roc = roc_auc_score(test_labels, predictions)
        
        self.metrics['gnn'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }

    def get_network_suspiciousness_score(self, user_id, graph):
        """
        Calculate network suspiciousness score for a user
        """
        with torch.no_grad():
            user_features = self._get_user_features(user_id, graph)
            score = self.gnn_model(user_features)
        return score.item()

    def _convert_to_pytorch_geometric(self, graph, labels):
        """
        Convert networkx graph to PyTorch Geometric format
        """
        # Implementation details for conversion
        pass

    def _get_user_features(self, user_id, graph):
        """
        Extract user features from graph
        """
        # Implementation details for feature extraction
        pass

# Dashboard Integration Components
class DashboardComponents:
    def __init__(self, fraud_detection_system):
        self.fds = fraud_detection_system

    def get_high_risk_users(self, graph, threshold=0.8):
        """
        Identify high-risk users based on GNN analysis
        """
        high_risk_users = []
        for user_id in graph.nodes():
            risk_score = self.fds.get_network_suspiciousness_score(user_id, graph)
            if risk_score > threshold:
                high_risk_users.append({
                    'user_id': user_id,
                    'risk_score': risk_score,
                    'connections': list(graph.neighbors(user_id))
                })
        return high_risk_users

    def get_performance_metrics(self):
        """
        Get comparative performance metrics for dashboard display
        """
        return self.fds.metrics

    def visualize_user_connections(self, user_id, graph):
        """
        Generate visualization data for user connections
        """
        subgraph = nx.ego_graph(graph, user_id, radius=2)
        return {
            'nodes': list(subgraph.nodes()),
            'edges': list(subgraph.edges()),
            'risk_scores': {
                node: self.fds.get_network_suspiciousness_score(node, graph)
                for node in subgraph.nodes()
            }
        }

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    
    return correct / total

def create_subgraph(data, mask):
    # Get the nodes in this split
    split_nodes = torch.where(mask)[0]
    
    # Create a mapping from old to new indices
    node_mapper = torch.full((data.x.size(0),), -1, dtype=torch.long)
    node_mapper[split_nodes] = torch.arange(len(split_nodes))
    
    # Get edges where both nodes are in the split
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    split_edges = data.edge_index[:, edge_mask]
    
    # Remap node indices
    split_edges = node_mapper[split_edges]
    
    # Get edge attributes for the split
    split_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
    
    return Data(
        x=data.x[mask],
        edge_index=split_edges,
        edge_attr=split_edge_attr,
        y=data.y[mask]
    )

class FraudDetectionAgent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transaction_history = deque(maxlen=100)
        self.fraud_patterns = {
            'fraud_rate': 0.0,
            'avg_confidence': 0.0,
            'trend': 'stable'
        }

    def detect_fraud(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            conf = F.softmax(out, dim=1).max(dim=1)[0]
            
            # Store transaction results for each prediction in the batch
            for p, c in zip(pred, conf):
                self.transaction_history.append({
                    'prediction': p.item(),  # Now converting single elements
                    'confidence': c.item()
                })
            
            return pred, conf

    def analyze_patterns(self):
        # Calculate fraud rate
        fraud_count = sum(1 for t in self.transaction_history if t['prediction'] == 1)
        self.fraud_patterns['fraud_rate'] = fraud_count / len(self.transaction_history)
        
        # Calculate average confidence
        self.fraud_patterns['avg_confidence'] = np.mean([t['confidence'] for t in self.transaction_history])
        
        # Determine trend
        if len(self.transaction_history) >= 2:
            recent_rate = sum(1 for t in list(self.transaction_history)[-10:] if t['prediction'] == 1) / 10
            older_rate = sum(1 for t in list(self.transaction_history)[:-10] if t['prediction'] == 1) / (len(self.transaction_history) - 10)
            
            if recent_rate > older_rate * 1.1:
                self.fraud_patterns['trend'] = 'increasing'
            elif recent_rate < older_rate * 0.9:
                self.fraud_patterns['trend'] = 'decreasing'
            else:
                self.fraud_patterns['trend'] = 'stable'
        
        return self.fraud_patterns

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    data_loader = FraudDataLoader('transaction_data.csv')
    data = data_loader.load_data()
    
    # Get the number of nodes from features
    n = data.x.size(0)
    
    # Resize labels to match feature dimensions
    data.y = data.y[:n]  # Take only the first n labels to match feature dimensions
    
    # Create masks for splitting
    indices = torch.randperm(n)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Apply masks to create split datasets using the new function
    train_data = create_subgraph(data, train_mask)
    val_data = create_subgraph(data, val_mask)
    test_data = create_subgraph(data, test_mask)
    
    # Create model and agent
    print("Initializing model and agent...")
    model = CGNN(input_dim=data.x.size(1), hidden_dim=64, output_dim=2).to(device)
    agent = FraudDetectionAgent(model, device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=32)
    test_loader = DataLoader([test_data], batch_size=32)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    for epoch in range(50):
        # Train
        loss = train_model(model, train_loader, optimizer, device)
        
        # Evaluate
        val_acc = evaluate(model, val_loader, device)
        
        # Agent analysis
        for data in val_loader:
            pred, conf = agent.detect_fraud(data)
        
        patterns = agent.analyze_patterns()
        if patterns:
            print(f"Agent Analysis - Fraud Rate: {patterns['fraud_rate']:.2f}, "
                  f"Confidence: {patterns['avg_confidence']:.2f}, "
                  f"Trend: {patterns['trend']}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Test final model
    print("\nTesting final model...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')
    
    # Final agent analysis
    print("\nFinal Agent Analysis:")
    for data in test_loader:
        pred, conf = agent.detect_fraud(data)
    patterns = agent.analyze_patterns()
    if patterns:
        print(f"Fraud Rate: {patterns['fraud_rate']:.2f}")
        print(f"Average Confidence: {patterns['avg_confidence']:.2f}")
        print(f"Trend: {patterns['trend']}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    try:
        data = results_queue.get_nowait()
        return jsonify(data)
    except queue.Empty:
        return jsonify({
            'fraudRate': 0,
            'confidence': 0,
            'trend': 'stable',
            'epoch': 0,
            'loss': 0,
            'valAcc': 0
        })

def training_thread():
    main()  # Call your existing main function

if __name__ == '__main__':
    # Start training in a separate thread
    thread = threading.Thread(target=training_thread)
    thread.start()
    
    # Start Flask server
    app.run(debug=True, use_reloader=False)

# Add this near the top with other initializations
results_queue = queue.Queue()
# Remove this duplicate class
# class CompetitiveGNNModel:
#     def __init__(self, input_dim, hidden_dim=64, output_dim=2):
#         self.normal_gnn = CGNN(input_dim, hidden_dim, output_dim)
#         self.fraud_gnn = CGNN(input_dim, hidden_dim, output_dim)
#         self.mutual_info_weight = 0.1
        
#     def mutual_information_loss(self, normal_emb, fraud_emb):
#         joint = torch.mean(torch.sum(normal_emb * fraud_emb, dim=1))
#         marginal = torch.mean(torch.sum(normal_emb * torch.roll(fraud_emb, 1, 0), dim=1))
#         return joint - marginal

#     def forward(self, x, edge_index):
#         normal_out = self.normal_gnn(x, edge_index)
#         fraud_out = self.fraud_gnn(x, edge_index)
#         return normal_out, fraud_out

def create_subgraph(data, mask):
    # Get the nodes in this split
    split_nodes = torch.where(mask)[0]
    
    # Create a mapping from old to new indices
    node_mapper = torch.full((data.x.size(0),), -1, dtype=torch.long)
    node_mapper[split_nodes] = torch.arange(len(split_nodes))
    
    # Get edges where both nodes are in the split
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    split_edges = data.edge_index[:, edge_mask]
    
    # Remap node indices
    split_edges = node_mapper[split_edges]
    
    # Get edge attributes for the split
    split_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
    
    return Data(
        x=data.x[mask],
        edge_index=split_edges,
        edge_attr=split_edge_attr,
        y=data.y[mask]
    )

class FraudDetectionAgent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transaction_history = deque(maxlen=100)
        self.fraud_patterns = {
            'fraud_rate': 0.0,
            'avg_confidence': 0.0,
            'trend': 'stable'
        }

    def detect_fraud(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            conf = F.softmax(out, dim=1).max(dim=1)[0]
            
            # Store transaction results for each prediction in the batch
            for p, c in zip(pred, conf):
                self.transaction_history.append({
                    'prediction': p.item(),  # Now converting single elements
                    'confidence': c.item()
                })
            
            return pred, conf

    def analyze_patterns(self):
        if not self.transaction_history:
            return None
        
        # Calculate fraud rate
        fraud_count = sum(1 for t in self.transaction_history if t['prediction'] == 1)
        self.fraud_patterns['fraud_rate'] = fraud_count / len(self.transaction_history)
        
        # Calculate average confidence
        self.fraud_patterns['avg_confidence'] = np.mean([t['confidence'] for t in self.transaction_history])
        
        # Determine trend
        if len(self.transaction_history) >= 2:
            recent_rate = sum(1 for t in list(self.transaction_history)[-10:] if t['prediction'] == 1) / 10
            older_rate = sum(1 for t in list(self.transaction_history)[:-10] if t['prediction'] == 1) / (len(self.transaction_history) - 10)
            
            if recent_rate > older_rate * 1.1:
                self.fraud_patterns['trend'] = 'increasing'
            elif recent_rate < older_rate * 0.9:
                self.fraud_patterns['trend'] = 'decreasing'
            else:
                self.fraud_patterns['trend'] = 'stable'
        
        return self.fraud_patterns

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    data_loader = FraudDataLoader('transaction_data.csv')
    data = data_loader.load_data()
    
    # Get the number of nodes from features
    n = data.x.size(0)
    
    # Resize labels to match feature dimensions
    data.y = data.y[:n]  # Take only the first n labels to match feature dimensions
    
    # Create masks for splitting
    indices = torch.randperm(n)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Apply masks to create split datasets using the new function
    train_data = create_subgraph(data, train_mask)
    val_data = create_subgraph(data, val_mask)
    test_data = create_subgraph(data, test_mask)
    
    # Create model and agent
    print("Initializing model and agent...")
    model = CGNN(input_dim=data.x.size(1), hidden_dim=64, output_dim=2).to(device)
    agent = FraudDetectionAgent(model, device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=32)
    test_loader = DataLoader([test_data], batch_size=32)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    for epoch in range(50):
        # Train
        loss = train_model(model, train_loader, optimizer, device)
        
        # Evaluate
        val_acc = evaluate(model, val_loader, device)
        
        # Agent analysis
        for data in val_loader:
            pred, conf = agent.detect_fraud(data)
        
        patterns = agent.analyze_patterns()
        if patterns:
            print(f"Agent Analysis - Fraud Rate: {patterns['fraud_rate']:.2f}, "
                  f"Confidence: {patterns['avg_confidence']:.2f}, "
                  f"Trend: {patterns['trend']}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Test final model
    print("\nTesting final model...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')
    
    # Final agent analysis
    print("\nFinal Agent Analysis:")
    for data in test_loader:
        pred, conf = agent.detect_fraud(data)
    patterns = agent.analyze_patterns()
    if patterns:
        print(f"Fraud Rate: {patterns['fraud_rate']:.2f}")
        print(f"Average Confidence: {patterns['avg_confidence']:.2f}")
        print(f"Trend: {patterns['trend']}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    try:
        data = results_queue.get_nowait()
        return jsonify(data)
    except queue.Empty:
        return jsonify({
            'fraudRate': 0,
            'confidence': 0,
            'trend': 'stable',
            'epoch': 0,
            'loss': 0,
            'valAcc': 0
        })

def training_thread():
    main()  # Call your existing main function

if __name__ == '__main__':
    # Start training in a separate thread
    thread = threading.Thread(target=training_thread)
    thread.start()
    
    # Start Flask server
    app.run(debug=True, use_reloader=False)

@app.route('/check_transaction', methods=['POST'])
def check_transaction():
    data = request.get_json()
    transaction_id = data.get('transaction_id')
    amount = data.get('amount', 0)
    
    # Simple risk calculation based on amount
    risk_score = min(amount / 10000, 1.0)  # Higher amounts = higher risk
    is_safe = risk_score < 0.7  # 70% threshold
    
    return jsonify({
        'status': 'safe' if is_safe else 'suspicious',
        'risk_score': risk_score
    })
    