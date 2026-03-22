import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`📡 ${config.method.toUpperCase()} ${config.url}`, config.params || config.data);
    return config;
  },
  (error) => {
    console.error('❌ Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error(`❌ ${error.response.status} ${error.config.url}:`, error.response.data);
    } else if (error.request) {
      // Request was made but no response
      console.error('❌ No response from server:', error.request);
    } else {
      // Something else happened
      console.error('❌ Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// WebSocket service
class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;
    this.listeners = new Map();
    this.statusListeners = new Set();
    this.messageQueue = [];
    this.shouldReconnect = true;
  }

  connect() {
    // Clear any pending reconnect
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Reset reconnect flag
    this.shouldReconnect = true;

    try {
      console.log('🔌 Connecting to WebSocket...');
      this.ws = new WebSocket(`${WS_URL}/ws`);
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.reconnectAttempts = 0;
        this.notifyStatusListeners('connected');
        
        // Send any queued messages
        while (this.messageQueue.length > 0) {
          const msg = this.messageQueue.shift();
          this.send(msg);
        }
      };

      this.ws.onclose = (event) => {
        console.log('❌ WebSocket disconnected:', event.reason || 'No reason provided');
        this.notifyStatusListeners('disconnected');
        
        // Attempt to reconnect if we should
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          const delay = Math.min(3000 * Math.pow(2, this.reconnectAttempts), 30000);
          console.log(`🔄 Reconnecting in ${delay/1000}s... (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
          
          this.reconnectTimeout = setTimeout(() => {
            this.reconnectAttempts++;
            this.connect();
          }, delay);
        } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.log('❌ Max reconnection attempts reached');
          this.notifyStatusListeners('failed');
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
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };

    } catch (error) {
      console.error('❌ Failed to connect WebSocket:', error);
      this.notifyStatusListeners('error');
    }
  }

  handleMessage(data) {
    // Handle different message types
    if (data.type === 'connection') {
      console.log('🔌 WebSocket connection established');
      return;
    }
    
    if (data.type === 'prediction') {
      this.notifyListeners('prediction', data.payload);
    } else if (data.type === 'alert') {
      this.notifyListeners('alert', data.payload);
    } else if (data.type === 'status') {
      this.notifyListeners('status', data.payload);
    } else if (data.type === 'heartbeat') {
      // Just ignore heartbeats
      return;
    } else {
      console.log('📨 Received message:', data.type);
    }
  }

  // Event listeners
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return this;
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
    return this;
  }

  // Status listeners
  onStatusChange(callback) {
    this.statusListeners.add(callback);
    return this;
  }

  offStatusChange(callback) {
    this.statusListeners.delete(callback);
    return this;
  }

  // Notify listeners
  notifyListeners(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`❌ Error in ${event} listener:`, error);
        }
      });
    }
  }

  notifyStatusListeners(status) {
    this.statusListeners.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error('❌ Error in status listener:', error);
      }
    });
  }

  // Send message
  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      console.log('📤 WebSocket message sent:', data);
      return true;
    } else {
      console.warn('⚠️ WebSocket not connected, queueing message');
      this.messageQueue.push(data);
      return false;
    }
  }

  // Disconnect
  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
    
    this.reconnectAttempts = 0;
    this.messageQueue = [];
    console.log('🔌 WebSocket disconnected');
  }

  // Get connection status
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

  // Check if connected
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

// API endpoints
export const predict = async (params) => {
  try {
    const response = await api.post('/predict', params);
    return response.data;
  } catch (error) {
    console.error('❌ Predict API error:', error);
    throw error;
  }
};

export const getHistorical = async (params) => {
  try {
    const response = await api.get('/historical', { params });
    return response.data;
  } catch (error) {
    console.error('❌ Historical API error:', error);
    throw error;
  }
};

export const getModelMetrics = async () => {
  try {
    const response = await api.get('/model-metrics');
    return response.data;
  } catch (error) {
    console.error('❌ Model metrics API error:', error);
    throw error;
  }
};

export const getHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('❌ Health API error:', error);
    throw error;
  }
};

// Create alert socket function (for backward compatibility)
export const createAlertSocket = (onMessage, onClose) => {
  const ws = new WebSocket(`${WS_URL}/ws`);
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('❌ Failed to parse WebSocket message:', error);
    }
  };
  
  ws.onclose = (event) => {
    console.log('❌ Alert socket closed:', event.reason);
    onClose();
  };
  
  ws.onerror = (error) => {
    console.error('❌ Alert socket error:', error);
  };
  
  return ws;
};

// Create WebSocket singleton
export const wsService = new WebSocketService();

export default api;