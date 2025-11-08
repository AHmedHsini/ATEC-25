"""
ECHO MIND VAULT - Python Simulation
ATEC'25 CyberQuest: Mission Horizon
Securing TARS-9 AI System aboard HORIZON-1

This simulation demonstrates:
1. Echo Predictor - ML-based anomaly detection (IsolationForest)
2. Anomaly Orchestrator - Quarantine and failover logic
3. Vault Logger - Immutable blockchain with PoA consensus
4. Mesh Router - Network resilience and rerouting
"""

import numpy as np
import hashlib
import time
import json
from datetime import datetime
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional
import random

# ============================================================================
# 1. ECHO PREDICTOR - ML Anomaly Detection
# ============================================================================

class EchoPredictor:
    """
    Mirror AI that learns normal TARS-9 behavior and detects deviations.
    Uses IsolationForest for unsupervised anomaly detection.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Args:
            contamination: Expected proportion of anomalies (10%)
            random_state: For reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
            max_samples=256
        )
        self.is_trained = False
        self.threshold = 1.8  # Anomaly score threshold
        
    def train(self, normal_data: np.ndarray):
        """Train on normal telemetry data"""
        print("🧠 Training Echo Predictor on normal behavior...")
        self.model.fit(normal_data)
        self.is_trained = True
        print("✓ Training complete. Model ready for detection.\n")
        
    def predict(self, sample: np.ndarray) -> Tuple[float, bool]:
        """
        Predict if sample is anomalous.
        
        Returns:
            (anomaly_score, is_anomaly)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Get anomaly score from IsolationForest
        # Returns -1 for anomalies, 1 for normal
        prediction = self.model.predict(sample.reshape(1, -1))[0]
        decision_score = self.model.decision_function(sample.reshape(1, -1))[0]
        
        # Convert to interpretable score (higher = more anomalous)
        # IsolationForest decision_function: negative = anomaly, positive = normal
        # We invert and scale to match our threshold system
        if prediction == -1:
            # Detected as anomaly - create high score
            anomaly_score = 2.0 + abs(decision_score) * 3.0
        else:
            # Normal - create low score based on how normal it is
            anomaly_score = max(0, 1.5 - decision_score * 1.5)
        
        is_anomaly = anomaly_score > self.threshold
        
        return anomaly_score, is_anomaly


# ============================================================================
# 2. VAULT LOGGER - Immutable Blockchain with PoA
# ============================================================================

class VaultBlock:
    """Single block in the immutable chain"""
    
    def __init__(self, index: int, timestamp: str, event: str, 
                 data: Dict, prev_hash: str, authority: str):
        self.index = index
        self.timestamp = timestamp
        self.event = event
        self.data = data
        self.prev_hash = prev_hash
        self.authority = authority  # PoA: which node validated
        self.hash = self._calculate_hash()
        
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block contents"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "event": self.event,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "authority": self.authority
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def __repr__(self):
        return (f"Block #{self.index} | {self.event} | "
                f"Hash: {self.hash[:16]}... | Auth: {self.authority}")


class VaultLogger:
    """
    Immutable log using blockchain structure with Proof of Authority.
    Ensures incident traceability and tamper-proof records.
    """
    
    def __init__(self, authorities: List[str]):
        """
        Args:
            authorities: List of trusted node names for PoA consensus
        """
        self.chain: List[VaultBlock] = []
        self.authorities = authorities
        self.current_authority_idx = 0
        
        # Create genesis block
        self._create_genesis()
        
    def _create_genesis(self):
        """Initialize chain with genesis block"""
        genesis = VaultBlock(
            index=0,
            timestamp=datetime.now().isoformat(),
            event="SYSTEM_INIT",
            data={"message": "Echo Mind Vault initialized"},
            prev_hash="0" * 64,
            authority="GENESIS"
        )
        self.chain.append(genesis)
        print(f"🔐 Vault initialized with genesis block: {genesis.hash[:16]}...\n")
        
    def add_block(self, event: str, data: Dict) -> VaultBlock:
        """
        Add new block with round-robin PoA consensus.
        
        Args:
            event: Event type (e.g., "ANOMALY_DETECTED")
            data: Event data dictionary
            
        Returns:
            The newly created block
        """
        prev_block = self.chain[-1]
        authority = self.authorities[self.current_authority_idx]
        self.current_authority_idx = (self.current_authority_idx + 1) % len(self.authorities)
        
        new_block = VaultBlock(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            event=event,
            data=data,
            prev_hash=prev_block.hash,
            authority=authority
        )
        
        self.chain.append(new_block)
        return new_block
    
    def validate_chain(self) -> bool:
        """Verify chain integrity by checking all hash links"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Check hash link
            if current.prev_hash != previous.hash:
                print(f"❌ Chain broken at block {i}")
                return False
                
            # Verify block hash
            if current.hash != current._calculate_hash():
                print(f"❌ Block {i} hash tampered")
                return False
                
        return True
    
    def get_latest_blocks(self, n: int = 5) -> List[VaultBlock]:
        """Get the n most recent blocks"""
        return self.chain[-n:]


# ============================================================================
# 3. MESH ROUTER - Network Resilience
# ============================================================================

class MeshNode:
    """Represents a node in the mesh network"""
    
    def __init__(self, name: str, is_ai: bool = False):
        self.name = name
        self.status = "standby" if is_ai and "BACKUP" in name else "active"
        self.load = 85 if name == "TARS-9" else 0
        self.is_ai = is_ai
        
    def quarantine(self):
        """Isolate this node"""
        self.status = "quarantined"
        self.load = 0
        
    def activate(self):
        """Activate for failover"""
        self.status = "active"
        if self.is_ai:
            self.load = 85
            
    def restore(self):
        """Return to normal operation"""
        if self.is_ai and "TARS" in self.name:
            self.status = "active"
            self.load = 85
        elif self.is_ai:
            self.status = "standby"
            self.load = 0


class MeshRouter:
    """
    Resilient network that handles node failures and rerouting.
    Implements Zero-Trust policies during incidents.
    """
    
    def __init__(self):
        self.nodes: Dict[str, MeshNode] = {
            "TARS-9": MeshNode("TARS-9", is_ai=True),
            "BACKUP-AI": MeshNode("BACKUP-AI", is_ai=True),
            "NODE-ALPHA": MeshNode("NODE-ALPHA"),
            "NODE-BETA": MeshNode("NODE-BETA")
        }
        self.zero_trust_active = False
        
    def route_to(self, target: str) -> str:
        """
        Route traffic to target node, finding alternatives if needed.
        
        Returns:
            Name of the node handling the request
        """
        if self.nodes[target].status == "active":
            return target
            
        # Find backup AI node
        for name, node in self.nodes.items():
            if node.is_ai and name != target:
                return name
                
        # This shouldn't happen, but return backup as fallback
        return "BACKUP-AI"
    
    def quarantine_node(self, node_name: str):
        """Isolate a compromised node"""
        self.nodes[node_name].quarantine()
        self.zero_trust_active = True
        print(f"🚨 {node_name} quarantined | Zero-Trust policies enforced")
        
    def failover(self, from_node: str, to_node: str):
        """Execute failover from compromised to backup"""
        self.nodes[to_node].activate()
        print(f"🔄 Failover: {from_node} → {to_node} | Load transferred")
        
    def restore_node(self, node_name: str):
        """Restore node after validation"""
        self.nodes[node_name].restore()
        
        # Deactivate backup if TARS-9 is restored
        if node_name == "TARS-9":
            self.nodes["BACKUP-AI"].restore()
            self.zero_trust_active = False
            
        print(f"✅ {node_name} restored | All systems nominal")
        
    def get_status(self) -> Dict:
        """Get current network status"""
        return {name: {"status": node.status, "load": node.load} 
                for name, node in self.nodes.items()}


# ============================================================================
# 4. ANOMALY ORCHESTRATOR - Incident Response
# ============================================================================

class AnomalyOrchestrator:
    """
    Coordinates response to detected anomalies:
    - Quarantine compromised components
    - Execute failover procedures
    - Log incidents to vault
    - Manage mesh rerouting
    """
    
    def __init__(self, vault: VaultLogger, mesh: MeshRouter):
        self.vault = vault
        self.mesh = mesh
        self.incidents = []
        
    def handle_anomaly(self, score: float, sample_data: np.ndarray, 
                       anomaly_type: str = "UNKNOWN") -> Dict:
        """
        Execute full incident response pipeline.
        
        Returns:
            Incident report dictionary
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🚨 ANOMALY DETECTED | Score: {score:.3f} | Type: {anomaly_type}")
        print(f"{'='*70}")
        
        # Step 1: Quarantine TARS-9
        print("\n[1/4] QUARANTINE")
        self.mesh.quarantine_node("TARS-9")
        
        # Step 2: Failover to backup
        print("\n[2/4] FAILOVER")
        backup = self.mesh.route_to("TARS-9")
        self.mesh.failover("TARS-9", backup)
        
        # Step 3: Mesh rerouting
        print("\n[3/4] MESH REROUTING")
        print(f"🌐 Traffic redirected through NODE-ALPHA & NODE-BETA")
        
        # Step 4: Vault sealing
        print("\n[4/4] VAULT SEALING")
        response_time = (time.time() - start_time) * 1000  # ms
        
        incident_data = {
            "type": anomaly_type,
            "score": f"{score:.3f}",
            "severity": "CRITICAL" if score > 3.0 else "HIGH",
            "action": "QUARANTINE_TARS-9",
            "failover_target": backup,
            "response_time_ms": f"{response_time:.2f}",
            "sample_features": sample_data.tolist()[:5]  # First 5 features
        }
        
        block = self.vault.add_block("ANOMALY_DETECTED", incident_data)
        print(f"🔒 Incident sealed in block #{block.index}")
        print(f"⏱️  Total response time: {response_time:.2f}ms")
        
        self.incidents.append(incident_data)
        
        return incident_data
    
    def restore_system(self):
        """Restore system after validation"""
        print(f"\n{'='*70}")
        print(f"🔧 SYSTEM RECOVERY INITIATED")
        print(f"{'='*70}")
        
        print("\nRunning diagnostics on TARS-9...")
        time.sleep(0.5)  # Simulate diagnostics
        
        print("✓ Diagnostics passed")
        print("✓ Integrity verified")
        print("✓ Behavioral baseline restored")
        
        self.mesh.restore_node("TARS-9")
        
        self.vault.add_block("SYSTEM_RECOVERED", {
            "message": "TARS-9 validated and restored",
            "incidents_handled": len(self.incidents)
        })


# ============================================================================
# 5. MAIN SIMULATION SYSTEM
# ============================================================================

class EchoMindVault:
    """
    Complete ECHO MIND VAULT system integrating all components.
    """
    
    def __init__(self):
        print("="*70)
        print("🛰️  ECHO MIND VAULT - INITIALIZING")
        print("="*70)
        print("HORIZON-1 Mission | TARS-9 Monitoring System")
        print("The shadow that predicts, proves, and protects\n")
        
        # Initialize components
        self.echo = EchoPredictor()
        self.vault = VaultLogger(authorities=["NODE-1", "NODE-2", "NODE-3"])
        self.mesh = MeshRouter()
        self.orchestrator = AnomalyOrchestrator(self.vault, self.mesh)
        
        # Statistics
        self.stats = {
            "scans": 0,
            "anomalies": 0,
            "false_positives": 0,
            "response_times": []
        }
        
    def generate_training_data(self, n_samples: int = 1000) -> np.ndarray:
        """Generate synthetic 'normal' TARS-9 telemetry"""
        print("📊 Generating training data (normal behavior)...")
        
        # Features: CPU, Memory, Network, Decision_Latency, Command_Rate
        # Sensor_A-E, Nav_X/Y/Z, Life_Support_Status
        normal_data = np.random.normal(loc=[
            50,    # CPU %
            60,    # Memory %
            30,    # Network %
            100,   # Decision latency (ms)
            10,    # Commands per minute
            0.5, 0.5, 0.5, 0.5, 0.5,  # Sensors A-E
            0, 0, 1,  # Navigation vector
            1.0    # Life support (nominal)
        ], scale=[
            10, 15, 10, 20, 3,
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.2, 0.2, 0.2,
            0.05
        ], size=(n_samples, 14))
        
        print(f"✓ Generated {n_samples} normal samples\n")
        return normal_data
    
    def generate_test_sample(self, anomaly: bool = False) -> np.ndarray:
        """Generate a single test sample (normal or anomalous)"""
        if not anomaly:
            # Normal sample - matches training distribution
            return np.random.normal(loc=[
                50, 60, 30, 100, 10,
                0.5, 0.5, 0.5, 0.5, 0.5,
                0, 0, 1, 1.0
            ], scale=[10, 15, 10, 20, 3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05])
        else:
            # Anomalous sample - EXTREME deviations that IsolationForest will catch
            anomaly_types = [
                # CPU spike - way outside normal range
                [98, 60, 30, 100, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1.0],
                # Memory leak - critical levels
                [50, 99, 30, 100, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1.0],
                # Network exfiltration - massive spike
                [50, 60, 98, 100, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1.0],
                # Erratic decision-making - 5x normal latency
                [50, 60, 30, 550, 55, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 1.0],
                # Sensor anomalies - multiple sensors failing
                [50, 60, 30, 100, 10, 2.5, 0.05, 3.0, 0.02, 2.8, 0, 0, 1, 1.0],
                # Navigation drift - severe off-course
                [50, 60, 30, 100, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 8, -6, 0.1, 1.0],
                # Life support failure - critical low
                [50, 60, 30, 100, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 1, 0.1],
                # Combined attack - multiple systems compromised
                [95, 95, 85, 450, 45, 1.8, 0.1, 2.2, 0.15, 1.9, 4, -3, 0.3, 0.4]
            ]
            base = np.array(random.choice(anomaly_types))
            # Add small noise to make it realistic
            return base + np.random.normal(0, 1, size=14)
    
    def train_system(self):
        """Train Echo Predictor on normal data"""
        training_data = self.generate_training_data(1000)
        self.echo.train(training_data)
        
    def run_detection_cycle(self, n_cycles: int = 20, anomaly_rate: float = 0.3):
        """
        Run continuous detection cycles.
        
        Args:
            n_cycles: Number of detection cycles to run
            anomaly_rate: Probability of injecting an anomaly
        """
        print("="*70)
        print("🔍 STARTING DETECTION CYCLES")
        print("="*70)
        print(f"Cycles: {n_cycles} | Anomaly injection rate: {anomaly_rate*100}%\n")
        
        anomaly_names = [
            "UNAUTHORIZED_COMMAND",
            "DATA_EXFILTRATION", 
            "BEHAVIORAL_DRIFT",
            "MEMORY_CORRUPTION",
            "TIMING_ATTACK",
            "MODEL_POISONING"
        ]
        
        for cycle in range(1, n_cycles + 1):
            # Generate sample
            inject_anomaly = random.random() < anomaly_rate
            sample = self.generate_test_sample(anomaly=inject_anomaly)
            
            # Detect
            score, is_anomaly = self.echo.predict(sample)
            self.stats["scans"] += 1
            
            status_icon = "🔴" if is_anomaly else "🟢"
            print(f"\n{status_icon} Cycle {cycle}/{n_cycles} | Score: {score:.3f} | ", end="")
            
            if is_anomaly:
                self.stats["anomalies"] += 1
                anomaly_type = random.choice(anomaly_names)
                print(f"ANOMALY: {anomaly_type}")
                
                # Handle incident
                self.orchestrator.handle_anomaly(score, sample, anomaly_type)
                
                # Auto-recovery after 1 second
                time.sleep(1)
                self.orchestrator.restore_system()
                
            else:
                print("NOMINAL - All systems operational")
                if inject_anomaly:
                    self.stats["false_positives"] += 1
                    print("  ⚠️  Note: Anomaly was injected but not detected (false negative)")
            
            time.sleep(0.3)  # Slight delay between cycles
    
    def print_statistics(self):
        """Display final statistics"""
        print("\n" + "="*70)
        print("📊 MISSION STATISTICS")
        print("="*70)
        
        print(f"\nTotal Scans:           {self.stats['scans']}")
        print(f"Anomalies Detected:    {self.stats['anomalies']}")
        print(f"Detection Rate:        {self.stats['anomalies']/self.stats['scans']*100:.1f}%")
        
        print(f"\nVault Status:")
        print(f"  Total Blocks:        {len(self.vault.chain)}")
        print(f"  Chain Valid:         {'✓ YES' if self.vault.validate_chain() else '✗ NO'}")
        
        print(f"\nMesh Network Status:")
        for node, status in self.mesh.get_status().items():
            print(f"  {node:12} {status['status']:12} Load: {status['load']:3}%")
        
        print("\n" + "="*70)
        print("🛰️  ECHO MIND VAULT - Mission Complete")
        print("="*70)
        print("Predict • Prove • Protect")
        print("The shadow that guards the light\n")
    
    def export_vault(self, filename: str = "vault_export.json"):
        """Export vault chain to JSON file"""
        export_data = {
            "mission": "HORIZON-1",
            "system": "ECHO_MIND_VAULT",
            "export_time": datetime.now().isoformat(),
            "chain_valid": self.vault.validate_chain(),
            "blocks": []
        }
        
        for block in self.vault.chain:
            export_data["blocks"].append({
                "index": block.index,
                "timestamp": block.timestamp,
                "event": block.event,
                "data": block.data,
                "prev_hash": block.prev_hash,
                "hash": block.hash,
                "authority": block.authority
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Vault exported to {filename}")


# ============================================================================
# 6. RUN SIMULATION
# ============================================================================

def main():
    """Main simulation entry point"""
    
    # Initialize system
    system = EchoMindVault()
    
    # Train Echo Predictor
    system.train_system()
    
    # Run detection cycles
    system.run_detection_cycle(n_cycles=15, anomaly_rate=0.35)
    
    # Display statistics
    system.print_statistics()
    
    # Export vault for verification
    system.export_vault()
    
    # Demonstrate vault integrity
    print("\n🔍 Validating vault chain integrity...")
    if system.vault.validate_chain():
        print("✅ Chain integrity verified - all blocks valid")
        print("\nLatest 3 blocks:")
        for block in system.vault.get_latest_blocks(3):
            print(f"  {block}")
    else:
        print("❌ Chain compromised - integrity check failed")


if __name__ == "__main__":
    main()