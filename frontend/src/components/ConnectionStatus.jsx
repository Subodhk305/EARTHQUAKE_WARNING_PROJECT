import { useState, useEffect } from 'react';
import { wsService } from '../services/api';

export default function ConnectionStatus() {
  const [status, setStatus] = useState('disconnected');

  useEffect(() => {
    setStatus(wsService.getStatus());
    
    wsService.onStatusChange((newStatus) => {
      setStatus(newStatus);
    });

    wsService.connect();

    return () => {
      wsService.disconnect();
    };
  }, []);

  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return { color: 'bg-green-500', text: 'Online', pulse: true };
      case 'connecting':
        return { color: 'bg-yellow-500', text: 'Connecting...', pulse: true };
      case 'error':
        return { color: 'bg-red-500', text: 'Error', pulse: false };
      default:
        return { color: 'bg-gray-500', text: 'Offline', pulse: false };
    }
  };

  const config = getStatusConfig();

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 backdrop-blur-sm rounded-full border border-gray-700">
      <div className={`w-2 h-2 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`} />
      <span className="text-sm font-medium text-gray-300">{config.text}</span>
    </div>
  );
}