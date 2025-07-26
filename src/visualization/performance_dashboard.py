import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import logging
from src.visualization.plot_signals import SignalVisualizer

logger = logging.getLogger('PerformanceDashboard')

class TradingDashboard:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.register_callbacks()
        logger.info("Initialized Trading Dashboard")

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Pocket Option Trading Dashboard", style={'textAlign': 'center'}),
            
            dcc.Tabs(id="tabs", value='signals', children=[
                dcc.Tab(label='Trading Signals', value='signals'),
                dcc.Tab(label='Performance Metrics', value='performance'),
                dcc.Tab(label='Live Trading', value='live'),
            ]),
            
            html.Div(id='tabs-content'),
            
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # 1 minute
                n_intervals=0
            )
        ])

    def register_callbacks(self):
        """Register dashboard callbacks"""
        @self.app.callback(
            Output('tabs-content', 'children'),
            Input('tabs', 'value')
        )
        def render_tab(tab):
            if tab == 'signals':
                return self.render_signals_tab()
            elif tab == 'performance':
                return self.render_performance_tab()
            elif tab == 'live':
                return self.render_live_tab()

        @self.app.callback(
            Output('signal-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_signals(n):
            return self.update_signal_graph()

    def render_signals_tab(self):
        """Render signals tab content"""
        return html.Div([
            html.H3("Recent Trading Signals"),
            dcc.Dropdown(
                id='symbol-selector',
                options=[{'label': s, 'value': s} for s in self.trading_system.symbols],
                value=self.trading_system.symbols[0],
                style={'width': '200px'}
            ),
            dcc.Graph(id='signal-graph')
        ])

    def render_performance_tab(self):
        """Render performance tab content"""
        return html.Div([
            html.H3("Model Performance Metrics"),
            dcc.Graph(id='accuracy-chart'),
            dcc.Graph(id='confusion-matrix'),
            dcc.Graph(id='feature-importance')
        ])

    def render_live_tab(self):
        """Render live trading tab"""
        return html.Div([
            html.H3("Live Trading"),
            html.Div(id='live-trades'),
            html.Button('Start Trading', id='start-trading', n_clicks=0),
            html.Button('Stop Trading', id='stop-trading', n_clicks=0)
        ])

    def update_signal_graph(self):
        """Update signal graph with latest data"""
        # This would fetch real data in production
        symbol = self.trading_system.symbols[0]
        signals = self.trading_system.get_recent_signals(symbol)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(go.Candlestick(
            x=signals['df']['timestamp'],
            open=signals['df']['open'],
            high=signals['df']['high'],
            low=signals['df']['low'],
            close=signals['df']['close'],
            name='Price'
        ))
        
        # Add signals
        for signal in signals['signals']:
            fig.add_trace(go.Scatter(
                x=[signal['timestamp']],
                y=[signal['price']],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='triangle-up' if signal['direction'] == 'BUY' else 'triangle-down',
                    color='green' if signal.get('validation', {}).get('success', False) else 'red'
                ),
                name=f"{signal['signal_type']} - {signal['direction']}"
            ))
        
        fig.update_layout(
            title=f"{symbol} Trading Signals",
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def run(self, port=8050):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on port {port}")
        self.app.run_server(port=port)
