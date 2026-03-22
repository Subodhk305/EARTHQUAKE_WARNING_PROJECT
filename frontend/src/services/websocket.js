// frontend/src/services/websocket.js
class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.listeners = new Map();
    this.statusListeners = new Set();
    this.reconnectTimeout = null;
  }

  connect() {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }
    
    try {
      console.log('🔌 Connecting to WebSocket...');
      this.ws = new WebSocket(`${wsUrl}/ws`);
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.reconnectAttempts = 0;
        this.notifyStatusListeners('connected');
        
        // Send a ping to verify connection
        this.send({
          type: 'ping'
        });
      };

      this.ws.onclose = (event) => {
        console.log('❌ WebSocket disconnected:', event.reason || 'No reason provided');
        this.notifyStatusListeners('disconnected');
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          const delay = Math.min(3000 * Math.pow(2, this.reconnectAttempts), 30000);
          console.log(`🔄 Reconnecting in ${delay/1000}s... (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
          
          this.reconnectTimeout = setTimeout(() => {
            this.reconnectAttempts++;
            this.connect();
          }, delay);
        } else {
          console.log('Max reconnect attempts reached');
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.notifyStatusListeners('error');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.notifyStatusListeners('error');
    }
  }

  handleMessage(data) {
    console.log('📨 Received message:', data.type);
    
    // Handle different message types from backend
    switch (data.type) {
      case 'connection':
        console.log('Connection confirmed:', data.message);
        this.notifyListeners('connection', data);
        break;
        
      case 'prediction_result':
        console.log('Prediction received:', data.data);
        this.notifyListeners('prediction', data.data);
        break;
        
      case 'model_status':
        console.log('Model status:', data.models_ready);
        this.notifyListeners('model_status', data);
        break;
        
      case 'pong':
        console.log('Pong received');
        this.notifyListeners('pong', data);
        break;
        
      case 'error':
        console.error('Backend error:', data.error);
        this.notifyListeners('error', data);
        break;
        
      case 'ack':
        console.log('Acknowledgment:', data.status);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
        this.notifyListeners('message', data);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  onStatusChange(callback) {
    this.statusListeners.add(callback);
  }

  offStatusChange(callback) {
    this.statusListeners.delete(callback);
  }

  notifyListeners(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in listener for ${event}:`, error);
        }
      });
    }
  }

  notifyStatusListeners(status) {
    this.statusListeners.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error('Error in status listener:', error);
      }
    });
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      console.log('📤 WebSocket message sent:', data.type || 'unknown');
    } else {
      console.warn('WebSocket not connected, message not sent');
    }
  }

  requestPrediction(locationData) {
    this.send({
      type: 'prediction',
      location_name: locationData.location_name,
      latitude: locationData.latitude,
      longitude: locationData.longitude,
      radius_km: locationData.radius_km || 200,
      include_waveform: locationData.include_waveform || false
    });
  }

  subscribe(location) {
    this.send({
      type: 'subscribe',
      location: location
    });
  }

  getModelStatus() {
    this.send({
      type: 'model_status'
    });
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  getStatus() {
    if (!this.ws) return 'disconnected';
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'disconnecting';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }
}

// Create and export singleton instance
const wsService = new WebSocketService();
export default wsService;