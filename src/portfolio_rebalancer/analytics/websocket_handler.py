"""WebSocket handler for real-time analytics progress updates."""

import json
import logging
import asyncio
from typing import Dict, Set, Any, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from .async_processing import AsyncAnalyticsProcessor
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager for WebSocket connections and real-time updates."""
    
    def __init__(self, async_processor: AsyncAnalyticsProcessor):
        """
        Initialize WebSocket manager.
        
        Args:
            async_processor: Async analytics processor for task status
        """
        self.async_processor = async_processor
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.task_subscribers: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.grafana_subscribers: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.running = False
        self.update_interval = 2.0  # seconds
        self.grafana_update_interval = 5.0  # seconds for Grafana live data
        
        logger.info("WebSocket manager initialized")
    
    async def register_connection(self, websocket: WebSocketServerProtocol, task_id: str):
        """
        Register a WebSocket connection for task updates.
        
        Args:
            websocket: WebSocket connection
            task_id: Task ID to subscribe to
        """
        try:
            # Add to general connections
            if task_id not in self.connections:
                self.connections[task_id] = set()
            self.connections[task_id].add(websocket)
            
            # Add to task subscribers
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            self.task_subscribers[task_id].add(websocket)
            
            logger.info(f"Registered WebSocket connection for task {task_id}")
            
            # Send initial status
            await self.send_task_status(task_id, websocket)
            
        except Exception as e:
            logger.error(f"Failed to register WebSocket connection: {e}")
    
    async def unregister_connection(self, websocket: WebSocketServerProtocol, task_id: str):
        """
        Unregister a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            task_id: Task ID to unsubscribe from
        """
        try:
            # Remove from connections
            if task_id in self.connections:
                self.connections[task_id].discard(websocket)
                if not self.connections[task_id]:
                    del self.connections[task_id]
            
            # Remove from task subscribers
            if task_id in self.task_subscribers:
                self.task_subscribers[task_id].discard(websocket)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]
            
            logger.info(f"Unregistered WebSocket connection for task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to unregister WebSocket connection: {e}")
    
    async def send_task_status(self, task_id: str, websocket: Optional[WebSocketServerProtocol] = None):
        """
        Send task status update to WebSocket connections.
        
        Args:
            task_id: Task ID
            websocket: Specific WebSocket to send to (None for all subscribers)
        """
        try:
            # Get current task status
            status = self.async_processor.get_task_status(task_id)
            
            # Prepare message
            message = {
                'type': 'task_status',
                'task_id': task_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': status
            }
            
            message_json = json.dumps(message)
            
            # Send to specific websocket or all subscribers
            if websocket:
                await self.send_to_websocket(websocket, message_json)
            else:
                await self.broadcast_to_task_subscribers(task_id, message_json)
                
        except Exception as e:
            logger.error(f"Failed to send task status for {task_id}: {e}")
    
    async def send_to_websocket(self, websocket: WebSocketServerProtocol, message: str):
        """
        Send message to a specific WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            message: JSON message to send
        """
        try:
            await websocket.send(message)
        except ConnectionClosed:
            logger.debug("WebSocket connection closed during send")
        except WebSocketException as e:
            logger.error(f"WebSocket error during send: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending WebSocket message: {e}")
    
    async def broadcast_to_task_subscribers(self, task_id: str, message: str):
        """
        Broadcast message to all subscribers of a task.
        
        Args:
            task_id: Task ID
            message: JSON message to broadcast
        """
        if task_id not in self.task_subscribers:
            return
        
        # Get copy of subscribers to avoid modification during iteration
        subscribers = self.task_subscribers[task_id].copy()
        
        # Send to all subscribers
        for websocket in subscribers:
            await self.send_to_websocket(websocket, message)
    
    async def start_status_updates(self):
        """Start periodic status updates for all active tasks."""
        self.running = True
        logger.info("Started WebSocket status updates")
        
        while self.running:
            try:
                # Get all active task IDs
                active_task_ids = set(self.task_subscribers.keys())
                
                # Send status updates for each active task
                for task_id in active_task_ids:
                    await self.send_task_status(task_id)
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def stop_status_updates(self):
        """Stop periodic status updates."""
        self.running = False
        logger.info("Stopped WebSocket status updates")
    
    async def register_grafana_connection(self, websocket: WebSocketServerProtocol, portfolio_id: str):
        """
        Register a WebSocket connection for Grafana live data.
        
        Args:
            websocket: WebSocket connection
            portfolio_id: Portfolio ID to subscribe to
        """
        try:
            if portfolio_id not in self.grafana_subscribers:
                self.grafana_subscribers[portfolio_id] = set()
            self.grafana_subscribers[portfolio_id].add(websocket)
            
            logger.info(f"Registered Grafana WebSocket connection for portfolio {portfolio_id}")
            
            # Send initial data
            await self.send_grafana_live_data(portfolio_id, websocket)
            
        except Exception as e:
            logger.error(f"Failed to register Grafana WebSocket connection: {e}")
    
    async def unregister_grafana_connection(self, websocket: WebSocketServerProtocol, portfolio_id: str):
        """
        Unregister a Grafana WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            portfolio_id: Portfolio ID to unsubscribe from
        """
        try:
            if portfolio_id in self.grafana_subscribers:
                self.grafana_subscribers[portfolio_id].discard(websocket)
                if not self.grafana_subscribers[portfolio_id]:
                    del self.grafana_subscribers[portfolio_id]
            
            logger.info(f"Unregistered Grafana WebSocket connection for portfolio {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Failed to unregister Grafana WebSocket connection: {e}")
    
    async def send_grafana_live_data(self, portfolio_id: str, websocket: Optional[WebSocketServerProtocol] = None):
        """
        Send live portfolio data to Grafana WebSocket connections.
        
        Args:
            portfolio_id: Portfolio ID
            websocket: Specific WebSocket to send to (None for all subscribers)
        """
        try:
            # Generate live data (in real implementation, get from analytics service)
            current_time = datetime.utcnow()
            
            live_data = {
                'type': 'grafana_live_data',
                'portfolio_id': portfolio_id,
                'timestamp': current_time.isoformat(),
                'metrics': {
                    'portfolio_value': 100000 + (current_time.minute * 100),
                    'daily_return': 0.005 + (current_time.second - 30) * 0.0001,
                    'volatility': 0.15 + (current_time.second % 10) * 0.001,
                    'sharpe_ratio': 1.2 + (current_time.second % 20 - 10) * 0.01,
                    'beta': 1.0 + (current_time.second % 15 - 7) * 0.01,
                    'alpha': 0.02 + (current_time.second % 12 - 6) * 0.001
                }
            }
            
            message_json = json.dumps(live_data)
            
            # Send to specific websocket or all subscribers
            if websocket:
                await self.send_to_websocket(websocket, message_json)
            else:
                await self.broadcast_to_grafana_subscribers(portfolio_id, message_json)
                
        except Exception as e:
            logger.error(f"Failed to send Grafana live data for {portfolio_id}: {e}")
    
    async def broadcast_to_grafana_subscribers(self, portfolio_id: str, message: str):
        """
        Broadcast message to all Grafana subscribers of a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            message: JSON message to broadcast
        """
        if portfolio_id not in self.grafana_subscribers:
            return
        
        # Get copy of subscribers to avoid modification during iteration
        subscribers = self.grafana_subscribers[portfolio_id].copy()
        
        # Send to all subscribers
        for websocket in subscribers:
            await self.send_to_websocket(websocket, message)

    async def handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle incoming WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: WebSocket path (should contain task ID or portfolio ID)
        """
        connection_id = None
        connection_type = None
        
        try:
            # Extract connection info from path
            path_parts = path.strip('/').split('/')
            
            if len(path_parts) >= 3 and path_parts[0] == 'ws':
                connection_type = path_parts[1]  # 'task', 'grafana', etc.
                connection_id = path_parts[2]    # task_id or portfolio_id
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid WebSocket path. Expected format: /ws/{type}/{id}'
                }))
                return
            
            # Handle different connection types
            if connection_type == 'task':
                # Register for task updates
                await self.register_connection(websocket, connection_id)
                
                # Send welcome message
                await websocket.send(json.dumps({
                    'type': 'connected',
                    'connection_type': 'task',
                    'task_id': connection_id,
                    'message': f'Connected to task {connection_id} updates'
                }))
                
            elif connection_type == 'grafana':
                # Register for Grafana live data
                await self.register_grafana_connection(websocket, connection_id)
                
                # Send welcome message
                await websocket.send(json.dumps({
                    'type': 'connected',
                    'connection_type': 'grafana',
 
    
    async def handle_websocket_message(self, websocket: WebSocketServerProtocol, 
                                     task_id: str, data: Dict[str, Any]):
        """
        Handle incoming WebSocket message.
        
        Args:
            websocket: WebSocket connection
            task_id: Task ID
            data: Parsed message data
        """
        message_type = data.get('type')
        
        if message_type == 'get_status':
            # Send current status
            await self.send_task_status(task_id, websocket)
            
        elif message_type == 'cancel_task':
            # Cancel the task
            success = self.async_processor.cancel_task(task_id)
            await websocket.send(json.dumps({
                'type': 'task_cancelled' if success else 'cancel_failed',
                'task_id': task_id,
                'success': success
            }))
            
        elif message_type == 'ping':
            # Respond to ping
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            }))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket connection statistics.
        
        Returns:
            Connection statistics
        """
        total_connections = sum(len(connections) for connections in self.connections.values())
        
        return {
            'total_connections': total_connections,
            'active_tasks': len(self.task_subscribers),
            'tasks_with_subscribers': list(self.task_subscribers.keys()),
            'running': self.running,
            'update_interval': self.update_interval
        }


class WebSocketServer:
    """WebSocket server for analytics progress updates."""
    
    def __init__(self, async_processor: AsyncAnalyticsProcessor, 
                 host: str = "localhost", port: int = 8085):
        """
        Initialize WebSocket server.
        
        Args:
            async_processor: Async analytics processor
            host: Host to bind to
            port: Port to bind to
        """
        self.async_processor = async_processor
        self.host = host
        self.port = port
        self.manager = WebSocketManager(async_processor)
        self.server = None
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            # Start the WebSocket server
            self.server = await websockets.serve(
                self.manager.handle_websocket_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Start status update loop
            asyncio.create_task(self.manager.start_status_updates())
            
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise AnalyticsError(f"WebSocket server startup failed: {e}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        try:
            # Stop status updates
            await self.manager.stop_status_updates()
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            logger.info("WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Server statistics
        """
        return {
            'host': self.host,
            'port': self.port,
            'running': self.server is not None,
            'connections': self.manager.get_connection_stats()
        }