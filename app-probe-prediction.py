import base64
import os

import chess.svg
import dash_bootstrap_components as dbc
import datasets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import regex as re
import torch
from dash import Dash, Input, Output, State, callback_context, dcc, html
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel

from src.modeling.data_collation import VariableLenUciCollator
from src.modeling.probe import Probe
from src.modeling.uci_tokenizers import UciTileTokenizer
from src.modeling.uci_utils import uci_to_board, uci_to_pgn


class ProbeAnalysisApp:

    app: Dash
    tok: UciTileTokenizer
    llm: GPT2LMHeadModel
    probe: Probe
    collate: VariableLenUciCollator

    LLM_PATH = "austindavis/chessGPT_d12"
    PROBE_REPO_ID = "austindavis/chessGPT_d12-board-scoring-probe"

    def __init__(self):
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            assets_external_path="assets",
        )
        self.dataset = datasets.load_dataset("austindavis/lichess-uci-scored", split="train")
        self.tok = UciTileTokenizer(upper_promotions=True)
        self.llm = GPT2LMHeadModel.from_pretrained(self.LLM_PATH).requires_grad_(False).cuda()

        config_path = hf_hub_download(repo_id=self.PROBE_REPO_ID, filename="config.pt")
        state_dict_path = hf_hub_download(repo_id=self.PROBE_REPO_ID, filename="state_dict.pt")

        self.probe = Probe.from_pretrained(os.path.dirname(state_dict_path))
        self.init_collate()
        self._setup_layout()
        self._setup_callbacks()

    def run(self):
        self.app.run(jupyter_mode="external", debug=False)

    def init_collate(self):
        self.collate = VariableLenUciCollator(self.tok, self.llm, self.probe)

    def _setup_layout(self):

        def create_hidden_stores():
            return html.Div(
                [
                    dcc.Store(id="llm-update-trigger"),
                    dcc.Store(id="probe-update-trigger"),
                    dcc.Store(id="cached-game-data"),
                ]
            )

        def create_header():
            return html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col-md-10",
                        children=[
                            html.Div(
                                className="page-header",
                                children=[
                                    html.H1("Probe Inspector!"),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        className="col-md-2",
                        children=[html.Div(create_color_mode_switch())],
                        style={"margin-left": "auto", "margin-right": 0},
                    ),
                ],
            )

        def create_color_mode_switch():
            return html.Span(
                [
                    dbc.Label(className="fa fa-moon", html_for="light-dark-switch"),
                    dbc.Switch(
                        id="light-dark-switch",
                        value=True,
                        className="d-inline-block ms-1",
                        persistence=True,
                    ),
                    dbc.Label(className="fa fa-sun", html_for="light-dark-switch"),
                ]
            )

        def create_move_predictions_switch():
            return html.Span(
                [
                    dbc.Label(className="fa fa-square", html_for="toggle-moves-switch"),
                    dbc.Switch(
                        id="toggle-moves-switch",
                        value=True,
                        className="d-inline-block ms-1",
                        persistence=True,
                    ),
                    dbc.Label(
                        className="fa-solid fa-chess-board",
                        html_for="toggle-moves-switch",
                    ),
                    dbc.Label(
                        "probe layer:",
                        html_for="arrows-probe-layer-input",
                        style={"padding-left": "10px", "padding-right": "5px"},
                    ),
                    dcc.Input(
                        id="arrows-probe-layer-input",
                        type="number",
                        min=0,
                        max=12,
                        value=9,
                        step=1,
                    ),
                ]
            )

        def create_instructions():
            return html.Div(
                className="col-md-3",
                children=[
                    html.Dl(
                        children=[
                            html.Dt("How to Use"),
                            html.Dd(
                                "Enter a game index to load a chess game from the lichess-uci-scored dataset. Specify an alternate probe or LLM path using the text fields below."
                            ),
                            html.Dt("Navigating Moves"),
                            html.Dd(
                                "Use the slider below the chessboard or click on a heatmap move to jump to that position."
                            ),
                            html.Dt("Heatmaps"),
                            html.Dd(
                                "Visualize relative advantage for each player across moves and layers. Click anywhere in the heatmap to update the chess board, transcript, and eval bars."
                            ),
                            html.Dt("Board"),
                            html.Dd(
                                "The chess board displays positions from specific games; use the toggle to show or hide probe action-value scores. Two eval bars indicate which player is ahead—Stockfish on the left, Probe on the right."
                            ),
                        ]
                    ),
                    create_input_fields(),
                ],
            )

        def create_chess_board():
            return html.Div(
                className="col-md-6",
                style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                },
                children=[
                    html.Div(
                        [
                            create_move_predictions_switch(),
                            html.Div(
                                children=[
                                    html.Img(
                                        id="chess-board",
                                        style={
                                            "transition": "opacity 0.3s ease-in-out",
                                            "opacity": 1,
                                        },
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Graph(
                                                id="eval-bar",
                                                style={
                                                    "height": "90%",
                                                    "width": "20px",
                                                    "border": "1px solid white",
                                                    "borderRadius": "4px",
                                                    "padding": "2px",
                                                    "backgroundColor": "black",
                                                },
                                                config={"staticPlot": True, "displayModeBar": False},
                                                animate=True,
                                                animation_options={
                                                    "frame": {"duration": 300, "redraw": True},
                                                    "transition": {"duration": 300, "easing": "linear"},
                                                },
                                            ),
                                            html.Label("Y"),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flex-direction": "column",
                                            "height": "400px",
                                            "justify-content": "center",
                                            "align-items": "center",
                                        },
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Graph(
                                                id="eval-bar-predicted",
                                                style={
                                                    "height": "90%",
                                                    "width": "20px",
                                                    "border": "1px solid white",
                                                    "borderRadius": "4px",
                                                    "padding": "2px",
                                                    "backgroundColor": "black",
                                                },
                                                config={"staticPlot": True, "displayModeBar": False},
                                                animate=True,
                                                animation_options={
                                                    "frame": {"duration": 300, "redraw": True},
                                                    "transition": {"duration": 300, "easing": "linear"},
                                                },
                                            ),
                                            html.Label("Ŷ"),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flex-direction": "column",
                                            "height": "400px",
                                            "justify-content": "center",
                                            "align-items": "center",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "max-width": "500px",
                                    "height": "auto",
                                    "display": "flex",
                                },
                            ),
                            html.Div(
                                dcc.Slider(
                                    id="board-slider",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=0,
                                    included=False,
                                    marks={i: "" for i in range(0, 101, 5)},
                                    # dots=False,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                        "template": "Ply: {value}",
                                    },
                                ),
                                style={
                                    "width": "100%",
                                    "max-width": "400px",
                                    "height": "auto",
                                },
                            ),
                        ]
                    ),
                ],
            )

        def create_input_fields():
            return html.Div(
                className="col-md-3",
                children=[
                    html.Div(
                        children=[
                            html.Label("LLM 🤗 Hf Repository:"),
                            dcc.Input(
                                id="llm-path",
                                type="text",
                                value=self.LLM_PATH,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Probe 🤗 Hf Repository:"),
                            dcc.Input(
                                id="probe-path",
                                type="text",
                                value=self.PROBE_REPO_ID,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"margin-bottom": "10px"},
                    ),
                    html.Div(
                        children=[
                            html.Label(f"Game Index: (min: 0 max: {len(self.dataset):,})"),
                            dcc.Input(
                                id="idx-input",
                                type="number",
                                min=0,
                                max=len(self.dataset),
                                value=0,
                                step=1,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"margin-bottom": "10px"},
                    ),
                ],
                style={"width": "100%"},
            )

        def creat_transcript_field():
            return html.Div(
                className="col-md-3",
                children=[
                    html.Label("Transcript:  "),
                    html.A("(View on Lichess.org)", href="", target="_blank", id="lichess-url"),
                    html.Br(),
                    html.Pre(
                        id="transcript-list",
                        children="",
                        style={
                            "white-space": "pre-line",
                            "font-family": "monospace",
                        },
                    ),
                ],
                style={"margin-bottom": "10px"},
            )

        def create_heatmap_section(title, graph_id):
            return html.Div(
                className="col-md-6",
                children=[
                    html.H3(title, style={"textAlign": "center"}),
                    dcc.Graph(id=graph_id, style={"height": "250px"}),
                    html.Label(
                        "Click cells to select ply/layer and update board", style={"textAlign": "center"}
                    ),
                ],
            )

        def create_error_section():
            return html.Div(
                className="row",
                children=[
                    html.Div(className="col-md-2"),
                    html.Div(
                        className="col-md-8 d-flex",
                        children=[
                            dcc.Graph(id="heatmap-errors", style={"height": "250px", "flex": "1"}),
                            dcc.Slider(
                                id="clamp-errors-slider",
                                min=0.05,
                                max=1.0,
                                step=0.05,
                                value=1.0,
                                marks=None,
                                vertical=True,
                                verticalHeight="250",
                                tooltip={
                                    "always_visible": True,
                                    "template": "Clamp Error: {value}",
                                    "placement": "right",
                                },
                            ),
                        ],
                    ),
                    html.Div(className="col-md-2"),
                ],
            )

        self.app.layout = html.Div(
            className="container-fluid",
            children=[
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col-md-12",
                            children=[
                                create_hidden_stores(),
                                create_header(),
                                html.Div(
                                    className="row",
                                    children=[
                                        create_instructions(),
                                        create_chess_board(),
                                        creat_transcript_field(),
                                    ],
                                ),
                                html.Div(
                                    className="row",
                                    children=[
                                        create_heatmap_section(
                                            "Actual Evaluation (Y)",
                                            "heatmap-actuals",
                                        ),
                                        create_heatmap_section(
                                            "Predicted Evaluation (Ŷ)",
                                            "heatmap-predictions",
                                        ),
                                    ],
                                ),
                                create_error_section(),
                            ],
                        )
                    ],
                ),
            ],
        )

    def interleave_arrays(self, arr1, arr2):
        L = arr1.shape[0]
        N, M = arr1.shape[1], arr2.shape[1]
        interleave_length = max(2 * N, M)  # TODO check this logic

        result = np.empty((L, interleave_length), dtype=arr1.dtype)

        # Assign even indices from arr1
        result[:, : 2 * M : 2] = arr1[:, :M]  # TODO check this logic Ensure arr1 is limited to M elements

        # Assign odd indices from arr2
        result[:, 1 : 2 * M : 2] = arr2  # Fully insert arr2 at odd indices

        # If N > M, insert the remaining elements from arr1
        if N > M:
            result[:, 2 * M :] = arr1[:, M:]  # Append leftover arr1 elements

        return result[:, : N + M]

    def maybe_convert_classes(self, preds, labels):
        """
        If a classification probe is used, convert
        class bin indices to percentages. This has no effect
        on a regression probe because nbins==1
        """
        nbins = float(next(iter(preds.values())).shape[-1])
        preds = {k: v.argmax(-1) / nbins for (k, v) in preds.items()}
        labels = {k: v / nbins for (k, v) in labels.items()}
        return preds, labels

    def format(self, dt, mode, neg=False):
        vals = torch.stack([dt[(i, mode)] for i in range(13)]).cpu().squeeze(-1).numpy()
        return 1 - vals if neg else vals

    def compute_errors(self, preds, lbl, M, neg):
        errs = self.format(preds, M, neg) - self.format(lbl, M, neg)
        return errs

    def get_heatmap_data(self, game_idx):
        record = self.dataset[game_idx]
        data = self.collate(self.dataset[game_idx])  # Collate single game
        inputs, labels = data.get(game=0)
        preds = self.probe.full_forward(inputs)

        preds, labels = self.maybe_convert_classes(preds, labels)

        predictions = self.interleave_arrays(
            self.format(preds, "white", neg=True),
            self.format(preds, "black"),
        )
        actuals = self.interleave_arrays(
            self.format(labels, "white"),
            self.format(labels, "black", neg=True),
        )

        errors = predictions - actuals

        moves = data.metadata["transcripts"][0].split()
        return errors, predictions, actuals, moves, record, preds

    def create_heatmap(self, data, moves, **kwargs):
        template = None
        if "template" in kwargs:
            template = kwargs.pop("template")
        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=[f"{i:02}: {mv}" for i, mv in enumerate(moves)],
                **kwargs,
            )
        )
        fig.update_layout(
            xaxis_title="Token",
            yaxis_title="Layer",
            margin=dict(l=10, r=10, t=30, b=10),
            template=template,
        )
        return fig

    def _setup_callbacks(self):

        @self.app.callback(
            Output("llm-update-trigger", "data"),  # This triggers the next callback
            Input("llm-path", "value"),
        )
        def update_llm(llm_path):
            self.llm = GPT2LMHeadModel.from_pretrained(llm_path).requires_grad_(False).cuda()
            self.init_collate()
            return {"trigger": True}

        @self.app.callback(
            Output("probe-update-trigger", "data"),
            Input("probe-path", "value"),
        )
        def update_probe(probe_path):
            config_path = hf_hub_download(repo_id=probe_path, filename="config.pt")
            state_dict_path = hf_hub_download(repo_id=probe_path, filename="state_dict.pt")
            self.probe = Probe.from_pretrained(os.path.dirname(state_dict_path)).train(False).cuda()
            self.init_collate()
            return {"trigger": True}

        self.app.clientside_callback(
            """
            (switchOn) => {
            document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark"); 
            return window.dash_clientside.no_update
            }
            """,
            Output("light-dark-switch", "id"),
            Input("light-dark-switch", "value"),
        )

        @self.app.callback(
            Output("heatmap-errors", "figure"),
            Output("heatmap-predictions", "figure"),
            Output("heatmap-actuals", "figure"),
            Input("cached-game-data", "data"),
            Input("clamp-errors-slider", "value"),
            Input("light-dark-switch", "value"),
            prevent_initial_call=True,
        )
        def update_heatmaps(cached_data, zscale, theme):
            errors = np.array(cached_data["errors"])
            predictions = np.array(cached_data["predictions"])
            actuals = np.array(cached_data["actuals"])
            moves = cached_data["moves"]

            template = {True: "plotly_white", False: "plotly_dark"}[theme]

            heatmap_predictions = self.create_heatmap(
                predictions,
                moves,
                showscale=True,
                colorscale="greys",
                zmid=0.5,
                template=template,
            )
            heatmap_actuals = self.create_heatmap(
                actuals[0:1],
                moves,
                y=["L "],
                showscale=True,
                colorscale="greys",
                zmid=0.5,
                template=template,
            )

            heatmap_errors = self.create_heatmap(
                errors,
                moves,
                showscale=True,
                zmin=-zscale,
                zmax=zscale,
                colorscale="RdBu",
                template=template,
            )

            return (
                heatmap_errors,
                heatmap_predictions,
                heatmap_actuals,
            )

        @self.app.callback(
            Output("cached-game-data", "data"),
            Output("board-slider", "max"),
            Output("board-slider", "marks"),
            Input("idx-input", "value"),
            Input("llm-update-trigger", "data"),
            Input("probe-update-trigger", "data"),
            # prevent_initial_call=True,
        )
        def handle_game_update(idx, llm_trigger, probe_trigger):
            errors, predictions, actuals, moves, record, preds = self.get_heatmap_data(idx)
            cache_data = {
                "errors": errors.tolist(),
                "predictions": predictions.tolist(),
                "actuals": actuals.tolist(),
                "moves": moves,
                "record": record,
            }

            moves = record["Transcript"].split()
            num_moves = len(moves)
            marks = {
                i: dict(label=f"{i}", style={"transform": "rotate(90deg)"}) for i, mv in enumerate(moves)
            }  # Generate slider marks

            return cache_data, num_moves - 1, marks

        @self.app.callback(
            Output("board-slider", "value"),
            Output("arrows-probe-layer-input", "value"),
            Input("heatmap-errors", "clickData"),
            Input("heatmap-predictions", "clickData"),
            Input("heatmap-actuals", "clickData"),
            State("arrows-probe-layer-input", "value"),
            prevent_initial_call=True,
        )
        def update_clicked_cell(click_errors, click_predictions, click_actuals, current_probe_layer_index):

            ctx = callback_context  # Get the callback context
            if not ctx.triggered:
                return None  # No input triggered, return default

            # Get the ID of the triggered input
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Determine the latest click data
            click_data = {
                "heatmap-errors": click_errors,
                "heatmap-predictions": click_predictions,
                "heatmap-actuals": click_actuals,
            }.get(trigger_id)

            move_index = layer = None
            if click_data:
                x_label = click_data["points"][0]["x"]
                y_label = click_data["points"][0]["y"]
                move_index = int(x_label.split(":")[0]) if x_label else None
                try:
                    layer = int(y_label)
                except:
                    layer = current_probe_layer_index

            return move_index, layer

        @self.app.callback(
            Output("chess-board", "src"),
            Output("lichess-url", "href"),
            Output("transcript-list", "children"),
            Output("eval-bar", "figure"),
            Output("eval-bar-predicted", "figure"),
            Input("cached-game-data", "data"),
            Input("toggle-moves-switch", "value"),
            Input("arrows-probe-layer-input", "value"),
            Input("board-slider", "value"),
            Input("light-dark-switch", "value"),
            prevent_initial_call=True,
        )
        def update_board(cached_data, show_moves, probe_layer, move_idx, theme):
            dark_theme = {
                "square light": "#7d828a",
                "square dark": "#2e343e",
                "square light lastmove": "#3e665c",
                "square dark lastmove": "#65a697",
            }
            light_theme = {
                "square light": "#f0d9b5",
                "square dark": "#b58863",
                "square light lastmove": "#2f709e",
                "square dark lastmove": "#265b80",
            }
            template = {True: light_theme, False: dark_theme}[theme]

            record = cached_data["record"]
            transcript = record["Transcript"]
            moves = transcript.split()[: move_idx + 1]
            sub_game = " ".join(moves).lower()
            board = uci_to_board(sub_game, fail_silent=True)

            if show_moves:
                arrows = self.get_move_arrows(cached_data, probe_layer, move_idx)
            else:
                arrows = []

            svg_data = chess.svg.board(
                board=board,
                size=400,
                arrows=arrows,
                lastmove=board.peek() if len(board.move_stack) else None,
                colors=template,
            )
            svg_bytes = svg_data.encode("utf-8")  # Convert SVG to base64
            base64_svg = base64.b64encode(svg_bytes).decode("utf-8")

            board_img_html = f"data:image/svg+xml;base64,{base64_svg}"

            formatted_transcript = self.format_transcript_for_display(sub_game, record)

            match = re.search(r'\[Site\s+"([^"]+)"\]', formatted_transcript)
            site_value = match.group(1) if match else ""
            url = f"https://lichess.org/{site_value}"

            move_score = 1 - np.array(cached_data["actuals"])[0, move_idx]
            eval_bar = self.get_eval_bar(move_score)

            predictions = 1 - np.array(cached_data["predictions"])[probe_layer, move_idx]
            eval_bar_pred = self.get_eval_bar(predictions)

            return (
                board_img_html,
                url,
                formatted_transcript,
                eval_bar,
                eval_bar_pred,
            )

    def get_eval_bar(self, move_score):
        eval_bar = go.Figure(
            data=go.Bar(
                x=["Y"],
                y=[move_score],
                marker_color="#ffffff",
                width=[1.0],
            )
        )

        eval_bar.update_layout(
            plot_bgcolor="#888888",
            paper_bgcolor="#888888",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-0.5, 0.5],
                fixedrange=True,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=True,
                range=[0, 1],
                fixedrange=True,
                tickmode="linear",
                tick0=0,
                dtick=0.1,
                ticks="inside",
                ticklen=20,
                tickwidth=1,
                tickcolor="black",
            ),
            bargap=0,
        )
        return eval_bar

    def get_move_arrows(self, cached_data, probe_layer, move_idx):
        board = uci_to_board(" ".join(cached_data["moves"][:move_idx]))
        scored_moves = self.score_moves(board, probe_layer)
        arrows = self.create_arrows(scored_moves)
        return arrows

    def score_moves(self, board, layer):
        mode = "white" if board.turn else "black"
        legal_moves = [m for m in board.generate_legal_moves()]
        transcript_prefix = " ".join([m.uci() for m in board.move_stack])
        scored_moves = []
        for move in legal_moves:
            transcript = " ".join([transcript_prefix, move.uci()]).strip()
            data = self.collate(transcript)
            single_game_hid_state = data.get("inputs", game=0, layer=layer, mode=mode)
            score_predictions = self.probe.forward(single_game_hid_state, layer, mode)
            num_bins = score_predictions.shape[-1]
            bins = torch.tensor(range(num_bins)).to(self.probe.device)
            weighted_score = ((score_predictions[-1].softmax(0) * bins).sum() / float(num_bins)).item()
            scored_moves.append((move, weighted_score))
        scored_moves.sort(key=lambda e: e[1])
        return scored_moves

    def create_arrows(self, predictions: list[tuple[chess.Move, float]]):
        alpha = 0.4

        def color(value):
            cmap = plt.get_cmap("viridis")  # Choose a divergent colormap
            r, g, b, a = cmap(value)
            # return r,g,b, alpha # alpha is lighter for middling values
            return (
                r,
                g,
                b,
                alpha,
            )  # alpha*abs(value-0.5)/0.5 # alpha is lighter for middling values

        radius = 0.5
        center = 0.5
        clamp = lambda v: max(center - radius, min(center + radius, v))
        clamped_predictions = [(m, clamp(s)) for (m, s) in predictions]
        min_score = min(clamped_predictions, key=lambda p: p[1])[1] - 1e-8
        max_score = max(clamped_predictions, key=lambda p: p[1])[1]
        normalized_predictions = [
            (m, (s - min_score) / (max_score - min_score)) for (m, s) in clamped_predictions
        ]

        arrows = [
            chess.svg.Arrow(m.from_square, m.to_square, color=ProbeAnalysisApp.rgb2hex(*color(s)))
            for (m, s) in sorted(normalized_predictions, key=lambda p: p[1])
            # sorted by score
        ]
        # best arrow gets alpha=100%
        arrows[-1].color = arrows[-1].color[:-2] + "ff"
        return arrows

    @staticmethod
    def rgb2hex(r, g, b, a):
        return "#%02x%02x%02x%02x" % (
            int(255 * r),
            int(255 * g),
            int(255 * b),
            int(255 * a),
        )

    def filter_pgn(self, pgn: str) -> str:
        return "\n".join(line for line in pgn.splitlines() if "?" not in line and "Result" not in line)

    def format_transcript_for_display(self, sub_game, record):
        headers = dict(
            Event=record["Event"],
            Site=record["Site"],
            Termination=record["Termination"],
        )
        pgn_repr = str(uci_to_pgn(sub_game, headers))
        return self.filter_pgn(pgn_repr)[:-1]


if __name__ == "__main__":
    ProbeAnalysisApp().run()
