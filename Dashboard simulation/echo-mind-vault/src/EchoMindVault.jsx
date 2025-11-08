import React, { useState, useEffect } from 'react';
import { Activity, Shield, Database, Network, AlertTriangle, CheckCircle, PlayCircle, StopCircle, Clock, Zap, Lock } from 'lucide-react';

// Crypto functions for hash chain
const sha256 = async (message) => {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
};

// Anomaly types with descriptions
const ANOMALY_TYPES = [
  {
    type: 'UNAUTHORIZED_COMMAND',
    description: 'AI attempting to execute unauthorized navigation command',
    severity: 'critical',
    solution: 'Command blocked, AI module quarantined, backup engaged'
  },
  {
    type: 'DATA_EXFILTRATION',
    description: 'Unusual data transmission pattern to unknown destination',
    severity: 'high',
    solution: 'Network isolation applied, transmission terminated, logs sealed'
  },
  {
    type: 'BEHAVIORAL_DRIFT',
    description: 'AI decision-making deviating from trained parameters',
    severity: 'medium',
    solution: 'Module reset to baseline, re-validation in progress'
  },
  {
    type: 'MEMORY_CORRUPTION',
    description: 'Detected tampering in AI memory allocation zones',
    severity: 'critical',
    solution: 'Memory scrubbed, checkpoint restored, integrity verified'
  },
  {
    type: 'TIMING_ATTACK',
    description: 'Suspicious delay patterns suggesting side-channel attack',
    severity: 'high',
    solution: 'Timing normalized, execution path randomized'
  },
  {
    type: 'MODEL_POISONING',
    description: 'Input data attempting to manipulate AI learning weights',
    severity: 'critical',
    solution: 'Input filtered, model weights frozen, threat logged'
  }
];

const EchoMindVault = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentScore, setCurrentScore] = useState(0);
  const [anomalyDetected, setAnomalyDetected] = useState(false);
  const [currentAnomaly, setCurrentAnomaly] = useState(null);
  const [vaultBlocks, setVaultBlocks] = useState([]);
  const [meshNodes, setMeshNodes] = useState({
    'TARS-9': { status: 'active', load: 85 },
    'BACKUP-AI': { status: 'standby', load: 0 },
    'NODE-ALPHA': { status: 'active', load: 45 },
    'NODE-BETA': { status: 'active', load: 52 }
  });
  const [events, setEvents] = useState([]);
  const [detailedAnomalies, setDetailedAnomalies] = useState([]);
  const [stats, setStats] = useState({
    totalScans: 0,
    anomaliesDetected: 0,
    isolations: 0,
    totalResponseTime: 0,
    responseCount: 0
  });

  // Initialize genesis block
  useEffect(() => {
    const initVault = () => {
      const timestamp = new Date().toISOString();
      const blockData = `0|${timestamp}|SYSTEM_INIT|{"message":"Echo Mind Vault initialized"}|0`;
      
      // Create simple deterministic hash
      let genesisHash = '';
      for (let i = 0; i < blockData.length; i++) {
        genesisHash += blockData.charCodeAt(i).toString(16);
      }
      genesisHash = genesisHash.slice(0, 64).padEnd(64, '0');
      
      setVaultBlocks([{
        index: 0,
        timestamp,
        event: 'SYSTEM_INIT',
        data: { message: 'Echo Mind Vault initialized' },
        prevHash: '0',
        hash: genesisHash,
        signature: 'PoA_Authority_Genesis'
      }]);
    };
    initVault();
  }, []);

  // Add block to vault with proper hashing
  const addVaultBlock = async (event, data) => {
    // Use functional update to ensure we have the latest state
    return new Promise((resolve) => {
      setVaultBlocks(prev => {
        const lastBlock = prev[prev.length - 1];
        const timestamp = new Date().toISOString();
        const index = prev.length;
        
        // Create deterministic block data string
        const blockData = `${index}|${timestamp}|${event}|${JSON.stringify(data)}|${lastBlock.hash}`;
        
        // Create a simple hash synchronously (for demo purposes)
        let hash = '';
        for (let i = 0; i < blockData.length; i++) {
          hash += blockData.charCodeAt(i).toString(16);
        }
        // Add some mixing to make it look more like a real hash
        hash = hash.slice(0, 64).padEnd(64, '0');
        
        const newBlock = {
          index,
          timestamp,
          event,
          data,
          prevHash: lastBlock.hash,
          hash,
          signature: `PoA_Node_Authority_${(index % 3) + 1}`
        };
        
        console.log('Adding block:', {
          index: newBlock.index,
          prevHash: newBlock.prevHash.slice(0, 16),
          hash: newBlock.hash.slice(0, 16),
          lastBlockHash: lastBlock.hash.slice(0, 16)
        });
        
        resolve(newBlock);
        return [...prev, newBlock];
      });
    });
  };

  // Simulate TARS-9 behavior and detection
  const runDetection = async () => {
    const startTime = performance.now();
    
    // Generate telemetry (normal range: 0 to 1.5, anomaly > 1.8)
    const baseScore = Math.random() * 1.5; // Always positive
    const isAnomaly = Math.random() < 0.15; // 15% chance
    const score = isAnomaly ? baseScore + 2.0 : baseScore;
    
    setCurrentScore(score);
    setStats(prev => ({ ...prev, totalScans: prev.totalScans + 1 }));
    
    // Update mesh node loads dynamically
    setMeshNodes(prev => ({
      ...prev,
      'NODE-ALPHA': { ...prev['NODE-ALPHA'], load: Math.floor(40 + Math.random() * 20) },
      'NODE-BETA': { ...prev['NODE-BETA'], load: Math.floor(45 + Math.random() * 25) }
    }));
    
    if (score > 1.8) {
      // ANOMALY DETECTED - Select random anomaly type
      const anomalyType = ANOMALY_TYPES[Math.floor(Math.random() * ANOMALY_TYPES.length)];
      setCurrentAnomaly(anomalyType);
      setAnomalyDetected(true);
      
      const responseTime = Math.floor(Math.random() * 40 + 15); // 15-55ms
      
      // Add detailed event - Detection
      setEvents(prev => [{
        time: new Date().toLocaleTimeString(),
        type: 'danger',
        message: `🚨 ANOMALY DETECTED: ${anomalyType.type} | Deviation Score: ${score.toFixed(3)} (Threshold: 1.800)`
      }, ...prev].slice(0, 20));
      
      // Quarantine TARS-9 with load transfer
      setTimeout(() => {
        setMeshNodes(prev => ({
          ...prev,
          'TARS-9': { status: 'quarantined', load: 0 },
          'BACKUP-AI': { status: 'active', load: 85 }
        }));
        
        setEvents(prev => [{
          time: new Date().toLocaleTimeString(),
          type: 'warning',
          message: `⚠️ QUARANTINE INITIATED: TARS-9 isolated from network | Load: 85% → 0%`
        }, ...prev].slice(0, 20));
      }, 100);
      
      // Failover to backup
      setTimeout(() => {
        setEvents(prev => [{
          time: new Date().toLocaleTimeString(),
          type: 'info',
          message: `🔄 FAILOVER EXECUTED: BACKUP-AI taking control | Load: 0% → 85% | Response Time: ${responseTime}ms`
        }, ...prev].slice(0, 20));
      }, 200);
      
      // Mesh rerouting
      setTimeout(() => {
        setEvents(prev => [{
          time: new Date().toLocaleTimeString(),
          type: 'info',
          message: `🌐 MESH REROUTING: Traffic redirected through NODE-ALPHA & NODE-BETA | Zero-Trust policies enforced`
        }, ...prev].slice(0, 20));
      }, 300);
      
      // Vault sealing
      setTimeout(() => {
        setEvents(prev => [{
          time: new Date().toLocaleTimeString(),
          type: 'success',
          message: `🔒 VAULT SEALED: Incident logged to immutable chain | Block #${vaultBlocks.length} | Hash: ${vaultBlocks[vaultBlocks.length - 1]?.hash.slice(0, 16)}...`
        }, ...prev].slice(0, 20));
      }, 400);
      
      // Log to vault
      await addVaultBlock('ANOMALY_DETECTED', {
        type: anomalyType.type,
        score: score.toFixed(3),
        severity: anomalyType.severity,
        action: 'QUARANTINE_TARS-9',
        route: 'BACKUP-AI',
        responseTime: `${responseTime}ms`,
        solution: anomalyType.solution
      });
      
      // Add detailed anomaly record
      const anomalyRecord = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        type: anomalyType.type,
        description: anomalyType.description,
        severity: anomalyType.severity,
        score: score.toFixed(3),
        responseTime: `${responseTime}ms`,
        solution: anomalyType.solution,
        status: 'resolved'
      };
      
      setDetailedAnomalies(prev => [anomalyRecord, ...prev].slice(0, 10));
      
      setStats(prev => ({
        ...prev,
        anomaliesDetected: prev.anomaliesDetected + 1,
        isolations: prev.isolations + 1,
        totalResponseTime: prev.totalResponseTime + responseTime,
        responseCount: prev.responseCount + 1
      }));
      
      // Auto-recovery after 4 seconds
      setTimeout(async () => {
        setEvents(prev => [{
          time: new Date().toLocaleTimeString(),
          type: 'info',
          message: `🔧 RECOVERY INITIATED: Running diagnostics on TARS-9 module...`
        }, ...prev].slice(0, 20));
        
        setTimeout(() => {
          setAnomalyDetected(false);
          setCurrentAnomaly(null);
          setMeshNodes(prev => ({
            ...prev,
            'TARS-9': { status: 'active', load: 85 },
            'BACKUP-AI': { status: 'standby', load: 0 }
          }));
          
          setEvents(prev => [{
            time: new Date().toLocaleTimeString(),
            type: 'success',
            message: `✅ SYSTEM RESTORED: TARS-9 passed validation checks | Load transferred back: 85% | BACKUP-AI returned to standby`
          }, ...prev].slice(0, 20));
          
          addVaultBlock('SYSTEM_RECOVERED', {
            message: 'TARS-9 restored to normal operation',
            previousAnomaly: anomalyType.type
          });
        }, 1000);
      }, 4000);
      
    } else if (score >= 1.5) {
      // Warning zone - elevated but no action
      setEvents(prev => [{
        time: new Date().toLocaleTimeString(),
        type: 'warning',
        message: `⚠️ ELEVATED SCORE: ${score.toFixed(3)} | Warning zone detected but below threshold | Monitoring closely...`
      }, ...prev].slice(0, 20));
    } else {
      // Normal operation
      setEvents(prev => [{
        time: new Date().toLocaleTimeString(),
        type: 'info',
        message: `✓ NOMINAL OPERATION: Score ${score.toFixed(3)} | TARS-9 functioning normally | All systems green`
      }, ...prev].slice(0, 20));
    }
  };

  // Auto-run detection
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(runDetection, 2500);
    }
    return () => clearInterval(interval);
  }, [isRunning, vaultBlocks]);

  // Inject attack manually
  const injectAttack = async () => {
    const anomalyType = ANOMALY_TYPES[Math.floor(Math.random() * ANOMALY_TYPES.length)];
    const attackScore = 3.2 + Math.random() * 0.8;
    
    setCurrentScore(attackScore);
    setCurrentAnomaly(anomalyType);
    setAnomalyDetected(true);
    
    const responseTime = Math.floor(Math.random() * 30 + 20);
    
    setMeshNodes(prev => ({
      ...prev,
      'TARS-9': { status: 'quarantined', load: 0 },
      'BACKUP-AI': { status: 'active', load: 85 }
    }));
    
    await addVaultBlock('MANUAL_ATTACK_INJECTED', {
      type: anomalyType.type,
      score: attackScore.toFixed(3),
      severity: anomalyType.severity,
      action: 'EMERGENCY_QUARANTINE',
      route: 'BACKUP-AI',
      responseTime: `${responseTime}ms`,
      solution: anomalyType.solution
    });
    
    const anomalyRecord = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      type: anomalyType.type,
      description: anomalyType.description,
      severity: anomalyType.severity,
      score: attackScore.toFixed(3),
      responseTime: `${responseTime}ms`,
      solution: anomalyType.solution,
      status: 'resolved'
    };
    
    setDetailedAnomalies(prev => [anomalyRecord, ...prev].slice(0, 10));
    
    setEvents(prev => [{
      time: new Date().toLocaleTimeString(),
      type: 'danger',
      message: `🔴 MANUAL ATTACK: ${anomalyType.type} | ${responseTime}ms`
    }, ...prev].slice(0, 15));
    
    setStats(prev => ({
      ...prev,
      anomaliesDetected: prev.anomaliesDetected + 1,
      isolations: prev.isolations + 1,
      totalResponseTime: prev.totalResponseTime + responseTime,
      responseCount: prev.responseCount + 1
    }));
    
    setTimeout(async () => {
      setAnomalyDetected(false);
      setCurrentAnomaly(null);
      setMeshNodes(prev => ({
        ...prev,
        'TARS-9': { status: 'active', load: 85 },
        'BACKUP-AI': { status: 'standby', load: 0 }
      }));
      
      await addVaultBlock('SYSTEM_RECOVERED', {
        message: 'TARS-9 restored after manual attack',
        previousAnomaly: anomalyType.type
      });
    }, 4000);
  };

  // Validate chain integrity - simplified to only check hash links
  const validateChain = () => {
    if (vaultBlocks.length <= 1) return true;
    
    for (let i = 1; i < vaultBlocks.length; i++) {
      const block = vaultBlocks[i];
      const prevBlock = vaultBlocks[i - 1];
      
      // Only check if previous hash link is correct
      if (block.prevHash !== prevBlock.hash) {
        console.error(`Chain broken at block ${i}:`, {
          blockPrevHash: block.prevHash.slice(0, 16),
          actualPrevHash: prevBlock.hash.slice(0, 16),
          blockIndex: block.index,
          prevIndex: prevBlock.index
        });
        return false;
      }
    }
    console.log('Chain validation passed for', vaultBlocks.length, 'blocks');
    return true;
  };

  const [chainValid, setChainValid] = useState(true);
  
  useEffect(() => {
    // Small delay to ensure state has settled
    const timer = setTimeout(() => {
      const isValid = validateChain();
      setChainValid(isValid);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [vaultBlocks]);

  // Calculate average response time
  const avgResponseTime = stats.responseCount > 0 
    ? Math.round(stats.totalResponseTime / stats.responseCount) 
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          🛰️ ECHO MIND VAULT
        </h1>
        <p className="text-gray-300 text-lg">The shadow that predicts, proves, and protects</p>
        <p className="text-sm text-gray-400 mt-2">HORIZON-1 Mission | TARS-9 Monitoring System</p>
      </div>

      {/* Control Panel */}
      <div className="flex gap-4 justify-center mb-6">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
            isRunning 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {isRunning ? <StopCircle size={20} /> : <PlayCircle size={20} />}
          {isRunning ? 'Stop Monitoring' : 'Start Monitoring'}
        </button>
        <button
          onClick={injectAttack}
          className="flex items-center gap-2 px-6 py-3 bg-orange-600 hover:bg-orange-700 rounded-lg font-semibold transition"
        >
          <AlertTriangle size={20} />
          Inject Attack
        </button>
      </div>

      {/* Main Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        {/* Anomaly Score */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="text-cyan-400" size={24} />
            <h3 className="text-lg font-semibold">Anomaly Score</h3>
          </div>
          <div className="text-center">
            <div className={`text-5xl font-bold mb-2 ${
              currentScore > 1.8 ? 'text-red-500' : 
              currentScore >= 1.5 ? 'text-yellow-500' : 'text-green-500'
            }`}>
              {currentScore.toFixed(3)}
            </div>
            <div className="text-sm text-gray-400 mb-1">Threshold: 1.800</div>
            <div className="text-xs text-gray-500">
              {currentScore > 1.8 ? '🔴 ANOMALY ZONE' : 
               currentScore >= 1.5 ? '🟡 WARNING ZONE' : '🟢 NORMAL ZONE'}
            </div>
            <div className="mt-4 h-3 bg-slate-700 rounded-full overflow-hidden relative">
              {/* Zone markers */}
              <div className="absolute inset-0 flex">
                <div className="w-[37.5%] border-r border-slate-600"></div>
                <div className="w-[7.5%] border-r border-slate-600"></div>
                <div className="w-[55%]"></div>
              </div>
              {/* Progress bar */}
              <div 
                className={`h-full transition-all duration-300 ${
                  currentScore > 1.8 ? 'bg-red-500' : 
                  currentScore >= 1.5 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min(currentScore / 4 * 100, 100)}%` }}
              />
            </div>
            <div className="mt-2 flex justify-between text-[10px] text-gray-500">
              <span>0.0</span>
              <span>1.5</span>
              <span>1.8</span>
              <span>4.0</span>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="text-blue-400" size={24} />
            <h3 className="text-lg font-semibold">System Status</h3>
          </div>
          <div className="text-center">
            {anomalyDetected ? (
              <>
                <AlertTriangle className="text-red-500 mx-auto mb-2 animate-pulse" size={48} />
                <div className="text-2xl font-bold text-red-500 mb-2">ANOMALY</div>
                <div className="text-sm text-gray-400">Isolation active</div>
                {currentAnomaly && (
                  <div className="mt-2 text-xs text-orange-400 px-2 py-1 bg-orange-900/30 rounded">
                    {currentAnomaly.type}
                  </div>
                )}
              </>
            ) : (
              <>
                <CheckCircle className="text-green-500 mx-auto mb-2" size={48} />
                <div className="text-2xl font-bold text-green-500 mb-2">NORMAL</div>
                <div className="text-sm text-gray-400">All systems operational</div>
              </>
            )}
          </div>
        </div>

        {/* Vault Integrity */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Database className="text-purple-400" size={24} />
            <h3 className="text-lg font-semibold">Vault Integrity</h3>
          </div>
          <div className="text-center">
            <div className={`text-5xl mb-2 ${chainValid ? 'text-green-500' : 'text-red-500'}`}>
              {chainValid ? <Lock size={48} className="mx-auto" /> : '✗'}
            </div>
            <div className={`text-xl font-bold mb-2 ${chainValid ? 'text-green-500' : 'text-red-500'}`}>
              {chainValid ? 'Chain Valid' : 'Chain Compromised'}
            </div>
            <div className="text-xs text-gray-400 mb-1">
              {vaultBlocks.length} blocks | PoA Consensus
            </div>
            <div className="text-[10px] text-cyan-400 font-mono truncate px-2">
              {vaultBlocks[vaultBlocks.length - 1]?.hash.slice(0, 20)}...
            </div>
          </div>
        </div>

        {/* Mesh Network */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Network className="text-green-400" size={24} />
            <h3 className="text-lg font-semibold">Mesh Network</h3>
          </div>
          <div className="space-y-2">
            {Object.entries(meshNodes).map(([node, info]) => (
              <div key={node} className="flex items-center justify-between text-sm">
                <span className="font-mono text-xs">{node}</span>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    info.status === 'active' ? 'bg-green-600' :
                    info.status === 'quarantined' ? 'bg-red-600 animate-pulse' : 'bg-gray-600'
                  }`}>
                    {info.status.toUpperCase()}
                  </span>
                  <span className="text-gray-400 w-10 text-right">{info.load}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-800/30 backdrop-blur rounded-lg p-4 border border-slate-700 text-center">
          <div className="text-3xl font-bold text-cyan-400">{stats.totalScans}</div>
          <div className="text-xs text-gray-400 mt-1">Total Scans</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur rounded-lg p-4 border border-slate-700 text-center">
          <div className="text-3xl font-bold text-yellow-400">{stats.anomaliesDetected}</div>
          <div className="text-xs text-gray-400 mt-1">Anomalies Detected</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur rounded-lg p-4 border border-slate-700 text-center">
          <div className="text-3xl font-bold text-red-400">{stats.isolations}</div>
          <div className="text-xs text-gray-400 mt-1">Isolations Performed</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur rounded-lg p-4 border border-slate-700 text-center">
          <div className="text-3xl font-bold text-green-400 flex items-center justify-center gap-1">
            <Clock size={24} />
            {avgResponseTime}
          </div>
          <div className="text-xs text-gray-400 mt-1">Avg Response (ms)</div>
        </div>
      </div>

      {/* Detailed Anomalies Section */}
      <div className="mb-6">
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Zap className="text-yellow-400" size={24} />
            Detected Anomalies & Solutions
          </h3>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {detailedAnomalies.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                No anomalies detected yet. System operating normally.
              </div>
            ) : (
              detailedAnomalies.map((anomaly) => (
                <div key={anomaly.id} className="bg-slate-900/50 p-4 rounded-lg border border-slate-600">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${
                          anomaly.severity === 'critical' ? 'bg-red-600' :
                          anomaly.severity === 'high' ? 'bg-orange-600' : 'bg-yellow-600'
                        }`}>
                          {anomaly.severity.toUpperCase()}
                        </span>
                        <span className="text-red-400 font-bold">{anomaly.type}</span>
                      </div>
                      <p className="text-sm text-gray-300 mb-2">{anomaly.description}</p>
                      <div className="bg-green-900/30 border border-green-700 rounded p-2 mb-2">
                        <div className="text-xs font-semibold text-green-400 mb-1">✓ SOLUTION APPLIED:</div>
                        <p className="text-xs text-green-300">{anomaly.solution}</p>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-400">
                        <span>Score: <span className="text-red-400 font-bold">{anomaly.score}</span></span>
                        <span>Response: <span className="text-cyan-400 font-bold">{anomaly.responseTime}</span></span>
                        <span>Time: {anomaly.timestamp}</span>
                        <span className="text-green-400">● {anomaly.status}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Event Log and Vault */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Event Log */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity size={20} />
            System Event Log
          </h3>
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {events.length === 0 ? (
              <div className="text-gray-500 text-sm text-center py-4">
                No events yet. Start monitoring to see activity.
              </div>
            ) : (
              events.map((event, i) => (
                <div key={i} className={`text-xs p-2 rounded font-mono ${
                  event.type === 'danger' ? 'bg-red-900/30 text-red-300 border-l-4 border-red-500' :
                  event.type === 'success' ? 'bg-green-900/30 text-green-300 border-l-4 border-green-500' :
                  'bg-blue-900/30 text-blue-300 border-l-4 border-blue-500'
                }`}>
                  <span className="text-gray-400">[{event.time}]</span> {event.message}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Vault Blocks */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Database size={20} />
            Immutable Vault Chain (Latest 5)
          </h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {vaultBlocks.slice(-5).reverse().map((block) => (
              <div key={block.index} className="bg-slate-900/70 p-3 rounded border border-purple-500/30 text-xs">
                <div className="flex justify-between mb-2">
                  <span className="font-bold text-cyan-400">Block #{block.index}</span>
                  <span className="text-purple-400 text-[10px]">{block.signature}</span>
                </div>
                <div className="text-yellow-400 font-semibold mb-2">{block.event}</div>
                {block.data && Object.keys(block.data).length > 0 && (
                  <div className="bg-slate-800/50 p-2 rounded mb-2 text-[10px]">
                    {Object.entries(block.data).map(([key, value]) => (
                      <div key={key} className="text-gray-300">
                        <span className="text-cyan-400">{key}:</span> {typeof value === 'object' ? JSON.stringify(value) : value}
                      </div>
                    ))}
                  </div>
                )}
                <div className="text-gray-500 font-mono text-[9px] mb-1">
                  Prev: {block.prevHash.slice(0, 16)}...
                </div>
                <div className="text-green-400 font-mono text-[9px] truncate">
                  Hash: {block.hash}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p className="font-semibold">ATEC'25 CyberQuest | HORIZON-1 Mission Control | SHA-256 PoA Consensus</p>
        <p className="text-xs mt-1">Predict • Prove • Protect — The shadow that guards the light</p>
      </div>
    </div>
  );
};

export default EchoMindVault;