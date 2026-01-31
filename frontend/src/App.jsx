import { useState, useEffect, useCallback } from 'react'

const API_BASE = '/api'

function App() {
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState('')
  const [latest, setLatest] = useState(null)
  const [detections, setDetections] = useState([])
  const [alerts, setAlerts] = useState([])
  const [analytics, setAnalytics] = useState(null)
  const [connected, setConnected] = useState(false)
  const [activeTab, setActiveTab] = useState('live')

  useEffect(() => {
    fetch(`${API_BASE}/cameras`)
      .then(r => r.json())
      .then(data => {
        setCameras(data.cameras || [])
        if (data.cameras?.length > 0 && !selectedCamera) {
          setSelectedCamera(data.cameras[0])
        }
        setConnected(true)
      })
      .catch(() => setConnected(false))
  }, [])

  useEffect(() => {
    if (!selectedCamera) return

    const poll = () => {
      fetch(`${API_BASE}/latest?camera_id=${selectedCamera}`)
        .then(r => r.json())
        .then(data => {
          if (data.item) setLatest(data.item)
          setConnected(true)
        })
        .catch(() => setConnected(false))

      fetch(`${API_BASE}/detections?camera_id=${selectedCamera}&limit=20`)
        .then(r => r.json())
        .then(data => setDetections(data.items || []))
        .catch(() => {})

      fetch(`${API_BASE}/alerts/recent`)
        .then(r => r.json())
        .then(data => setAlerts(data.alerts || []))
        .catch(() => {})

      fetch(`${API_BASE}/analytics/summary?camera_id=${selectedCamera}`)
        .then(r => r.json())
        .then(data => setAnalytics(data))
        .catch(() => {})
    }

    poll()
    const interval = setInterval(poll, 1000)
    return () => clearInterval(interval)
  }, [selectedCamera])

  const electronicsCount = latest?.objects?.filter(o => o.is_electronic).length || 0

  return (
    <div className="app">
      <header>
        <h1>🎯 Sentinel AI</h1>
        <div className="status">
          <span className={`status-dot ${connected ? '' : 'offline'}`}></span>
          {connected ? 'Connected' : 'Disconnected'}
          {electronicsCount > 0 && (
            <span className="alert-badge">⚠️ {electronicsCount} Electronics</span>
          )}
        </div>
      </header>

      <div className="tabs">
        <button className={activeTab === 'live' ? 'active' : ''} onClick={() => setActiveTab('live')}>
          Live Feed
        </button>
        <button className={activeTab === 'alerts' ? 'active' : ''} onClick={() => setActiveTab('alerts')}>
          Alerts {alerts.length > 0 && <span className="badge">{alerts.length}</span>}
        </button>
        <button className={activeTab === 'analytics' ? 'active' : ''} onClick={() => setActiveTab('analytics')}>
          Analytics
        </button>
      </div>

      {activeTab === 'live' && (
        <LiveFeed 
          cameras={cameras}
          selectedCamera={selectedCamera}
          setSelectedCamera={setSelectedCamera}
          latest={latest}
          detections={detections}
        />
      )}

      {activeTab === 'alerts' && <AlertsPanel alerts={alerts} />}
      
      {activeTab === 'analytics' && <AnalyticsDashboard analytics={analytics} camera={selectedCamera} />}
    </div>
  )
}


function LiveFeed({ cameras, selectedCamera, setSelectedCamera, latest, detections }) {
  const objectCounts = () => {
    if (!latest?.objects) return {}
    const counts = {}
    latest.objects.forEach(obj => {
      const label = obj.label || 'unknown'
      counts[label] = (counts[label] || 0) + 1
    })
    return counts
  }

  const counts = objectCounts()
  const totalObjects = Object.values(counts).reduce((a, b) => a + b, 0)
  const electronicsCount = latest?.objects?.filter(o => o.is_electronic).length || 0

  return (
    <div className="grid">
      <div className="card">
        <h2>Live Feed</h2>
        
        <div className="camera-select">
          <select value={selectedCamera} onChange={e => setSelectedCamera(e.target.value)}>
            {cameras.length === 0 && <option value="">No cameras</option>}
            {cameras.map(cam => (
              <option key={cam} value={cam}>{cam}</option>
            ))}
          </select>
        </div>

        <div className={`video-container ${latest?.has_electronics ? 'alert-border' : ''}`}>
          {latest?.image_data ? (
            <img src={`data:image/jpeg;base64,${latest.image_data}`} alt="Camera feed" />
          ) : (
            <span className="placeholder">
              {selectedCamera ? 'Waiting for frames...' : 'Select a camera'}
            </span>
          )}
        </div>

        <div className="stats">
          <div className="stat">
            <div className="stat-value">{totalObjects}</div>
            <div className="stat-label">Objects</div>
          </div>
          <div className={`stat ${electronicsCount > 0 ? 'alert' : ''}`}>
            <div className="stat-value">{electronicsCount}</div>
            <div className="stat-label">Electronics</div>
          </div>
          <div className="stat">
            <div className="stat-value">{latest?.sequence_number || 0}</div>
            <div className="stat-label">Frame #</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Detection History</h2>
        <div className="detections-list">
          {detections.length === 0 ? (
            <div className="empty-state">No detections yet</div>
          ) : (
            detections.map((det, i) => (
              <DetectionItem key={det.frame_id || i} detection={det} />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function DetectionItem({ detection }) {
  const objects = detection.objects || []
  const time = new Date(detection.timestamp_ms).toLocaleTimeString()
  const hasElectronics = objects.some(o => o.is_electronic)
  
  return (
    <div className={`detection-item ${hasElectronics ? 'has-electronics' : ''}`}>
      <div className="detection-header">
        <span>Frame #{detection.sequence_number}</span>
        <span>{time}</span>
      </div>
      <div className="detection-objects">
        {objects.length === 0 ? (
          <span className="object-tag">No objects</span>
        ) : (
          objects.map((obj, i) => (
            <span key={i} className={`object-tag ${obj.is_electronic ? 'electronic' : ''}`}>
              {obj.label}
            </span>
          ))
        )}
      </div>
    </div>
  )
}

function AlertsPanel({ alerts }) {
  return (
    <div className="card alerts-panel">
      <h2>🚨 Recent Alerts</h2>
      {alerts.length === 0 ? (
        <div className="empty-state">No alerts yet. Electronics will trigger alerts when detected.</div>
      ) : (
        <div className="alerts-list">
          {alerts.map((alert, i) => (
            <div key={i} className="alert-item">
              <div className="alert-header">
                <span className="alert-type">{alert.alert_type}</span>
                <span className="alert-camera">{alert.camera_id}</span>
                <span className="alert-time">
                  {new Date(alert.timestamp_ms).toLocaleString()}
                </span>
              </div>
              <div className="alert-objects">
                {(alert.objects || []).map((obj, j) => (
                  <span key={j} className="object-tag electronic">{obj.label}</span>
                ))}
              </div>
              {alert.clip_path && (
                <div className="alert-clip">📁 Clip saved</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


function AnalyticsDashboard({ analytics, camera }) {
  const [electronicsStats, setElectronicsStats] = useState(null)
  const [timeline, setTimeline] = useState([])

  useEffect(() => {
    if (!camera) return

    fetch(`/api/analytics/electronics?camera_id=${camera}&hours=24`)
      .then(r => r.json())
      .then(data => setElectronicsStats(data))
      .catch(() => {})

    fetch(`/api/analytics/timeline?camera_id=${camera}&minutes=60&bucket_minutes=5`)
      .then(r => r.json())
      .then(data => setTimeline(data.timeline || []))
      .catch(() => {})
  }, [camera])

  if (!analytics) {
    return <div className="card"><div className="empty-state">Loading analytics...</div></div>
  }

  const objectCounts = analytics.object_counts_last_hour || {}
  const sortedObjects = Object.entries(objectCounts).sort((a, b) => b[1] - a[1])
  const maxCount = Math.max(...Object.values(objectCounts), 1)

  return (
    <div className="analytics-grid">
      <div className="card">
        <h2>📊 Summary - {camera}</h2>
        <div className="stats-row">
          <div className="big-stat">
            <div className="big-stat-value">{analytics.total_frames}</div>
            <div className="big-stat-label">Total Frames</div>
          </div>
          <div className="big-stat">
            <div className="big-stat-value">{analytics.unique_objects?.length || 0}</div>
            <div className="big-stat-label">Object Types</div>
          </div>
          <div className="big-stat alert">
            <div className="big-stat-value">{electronicsStats?.total_electronics_detections || 0}</div>
            <div className="big-stat-label">Electronics (24h)</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>🔌 Electronics Detected (24h)</h2>
        {electronicsStats?.by_type && Object.keys(electronicsStats.by_type).length > 0 ? (
          <div className="bar-chart">
            {Object.entries(electronicsStats.by_type).sort((a, b) => b[1] - a[1]).map(([label, count]) => (
              <div key={label} className="bar-row">
                <span className="bar-label">{label}</span>
                <div className="bar-container">
                  <div 
                    className="bar electronic" 
                    style={{ width: `${(count / Math.max(...Object.values(electronicsStats.by_type))) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{count}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">No electronics detected in the last 24 hours</div>
        )}
      </div>

      <div className="card">
        <h2>📈 Objects Detected (Last Hour)</h2>
        {sortedObjects.length > 0 ? (
          <div className="bar-chart">
            {sortedObjects.slice(0, 10).map(([label, count]) => (
              <div key={label} className="bar-row">
                <span className="bar-label">{label}</span>
                <div className="bar-container">
                  <div className="bar" style={{ width: `${(count / maxCount) * 100}%` }} />
                </div>
                <span className="bar-value">{count}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">No detections in the last hour</div>
        )}
      </div>

      <div className="card wide">
        <h2>⏱️ Detection Timeline (Last Hour)</h2>
        {timeline.length > 0 ? (
          <div className="timeline-chart">
            {timeline.map((bucket, i) => (
              <div key={i} className="timeline-bar">
                <div 
                  className="timeline-fill" 
                  style={{ height: `${Math.min((bucket.total / 50) * 100, 100)}%` }}
                />
                <span className="timeline-label">
                  {new Date(bucket.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">No timeline data available</div>
        )}
      </div>
    </div>
  )
}

export default App
