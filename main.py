import sys
import os
import shutil
import json
import threading
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QStatusBar, QTextEdit,
    QFileDialog, QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QAbstractItemView, QMenu, QInputDialog, QMessageBox, QDockWidget, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDir, QSize, QTimer
from PyQt6.QtGui import QAction, QColor, QCursor


import qtawesome as qta
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import serial
    import pyeit.eit.jac as jac
    import pyeit.mesh as mesh
    from pyeit.eit.fem import EITForward
    from pyeit.eit.interp2d import sim2pts
    import pyeit.eit.protocol as protocol
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    MISSING_DEP = str(e)

PREFS_FILE = os.path.join(os.path.expanduser("~"), ".eit_tomographer_prefs.json")

def load_prefs() -> dict:
    try:
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_prefs(data: dict, workspace_folder: str = None):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    # Salva no workspace, dentro da pasta .eit
    if workspace_folder:
        try:
            eit_folder = os.path.join(workspace_folder, "config")
            os.makedirs(eit_folder, exist_ok=True)
            workspace_file = os.path.join(eit_folder, "eit_workspace.json")
            with open(workspace_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class EITWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    reference_done = pyqtSignal(np.ndarray)
    log_message    = pyqtSignal(str)
    error_signal   = pyqtSignal(str)
    finished_ok    = pyqtSignal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._running = False
        self._mode = "reference"
        self.utom = None
        self.v0 = None
        self.eit = None
        self.pts = None
        self.tri = None
        self.x = None
        self.y = None
        self.mesh_obj = None
        self.protocol_obj = None
        self.stimulation_pattern = None

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_message.emit(f"[{ts}] {msg}")

    def remove_excitation_measurements(self, data, window=3, threshold=2.0):
        data = data.astype(float)
        cleaned = data.copy()
        for i in range(len(data)):
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            neighbors = np.concatenate([data[start:i], data[i+1:end]])
            local_mean = np.mean(neighbors)
            local_std = np.std(neighbors)
            if abs(data[i] - local_mean) > threshold * local_std:
                cleaned[i] = local_mean
        return cleaned.astype(np.float32)

    def normalize_array(self, arr):
        arr_max = np.max(arr)
        return arr if arr_max == 0 else arr / arr_max

    def collect_measurements(self, avg):
        data = np.zeros(self.n_commands, dtype=np.float32)
        for i in range(self.n_commands):
            if not self._running:
                break
            channel = self.stimulation_pattern[i, :].astype(np.uint8)
            self.utom.write(bytearray([ord('S')]) + bytearray(channel))
            cmd_measure = bytearray([ord('F')]) + bytearray(channel)
            for _ in range(avg):
                if not self._running:
                    break
                self.utom.write(cmd_measure)
                raw = self.utom.read(4)
                if len(raw) == 4:
                    data[i] += np.frombuffer(raw, dtype=np.float32)[0] / avg
                else:
                    self.log(f"Sem resposta na posicao {i}")
            if not self._running:
                break

        if not self._running:
            self.log("Coleta interrompida — frame não será salvo.")
            return data

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        os.makedirs(self.config["data_folder"], exist_ok=True)
        np.savetxt(f"{self.config['data_folder']}/frame_{ts}.txt", data, fmt="%.6f")
        return data

    def setup_serial(self):
        self.log(f"Conectando em {self.config['port']} @ {self.config['baudrate']} baud...")
        self.utom = serial.Serial(
            port=self.config["port"], baudrate=self.config["baudrate"],
            bytesize=self.config["data_bits"], parity=self.config["parity"],
            stopbits=self.config["stop_bits"], timeout=1,   # era 50 (segundos!)
        )
        self.utom.write(bytearray([ord('E'), self.config["freq_sel"], 0, 0, 0]))
        self.utom.write(bytearray([ord('G'), self.config["gainDB"], self.config["gainLIN"], 0, 0]))
        self.log("Serial configurada.")

    def setup_eit(self):
        self.log("Configurando malha EIT...")
        n_el = self.config["n_el"]
        mode_idx = self.config.get("mode_idx", 0)
        dist_exc = 1 if mode_idx == 0 else n_el // 2
        
        self.mesh_obj = mesh.create(n_el, h0=self.config["h0"])
        self.pts = self.mesh_obj.node
        self.tri = self.mesh_obj.element
        self.x, self.y = self.pts[:, 0], self.pts[:, 1]
        self.protocol_obj = protocol.create(n_el, dist_exc=dist_exc, step_meas=1, parser_meas="std")
        EITForward(self.mesh_obj, self.protocol_obj).solve_eit()
        self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
        self.eit.setup(p=0.5, lamb=0.2, method="kotre", perm=1, jac_normalized=True)
        self.log(f"Malha EIT pronta ({n_el} eletrodos, modo {'adjacente' if mode_idx==0 else 'radial'}).")

    def load_stimulation_pattern(self):
        workspace = self.config["workspace"]
        eit_folder = os.path.join(workspace, "config")
        n_el = self.config["n_el"]
        mode_idx = self.config.get("mode_idx", 0)
        filename = f"stim_pattern_{n_el}_{mode_idx}.txt"
        filepath = os.path.join(eit_folder, filename)

        if os.path.exists(filepath):
            # Carrega o padrão do arquivo
            self.stimulation_pattern = np.loadtxt(filepath, delimiter=",").astype(np.uint8)
            self.n_commands = len(self.stimulation_pattern)
            self.log(f"Padrões carregados: {self.n_commands} comandos de {filepath}")
            
            # Verifica se o formato está correto (opcional)
            if self.stimulation_pattern.shape[1] != 4:
                self.log(f"Aviso: Padrão tem {self.stimulation_pattern.shape[1]} colunas, esperado 4")
        else:
            self.log(f"Arquivo de padrões não encontrado: {filepath}. Gerando...")
            os.makedirs(eit_folder, exist_ok=True)
            self._generate_and_save_pattern(filepath)

    def _generate_and_save_pattern(self, filepath, mode="adjacent"):
        """Gera e salva o padrão de estimulação (adjacent ou radial)."""
        n_el = self.config["n_el"]

        rows = []
        if mode == "adjacent":
            for inj in range(n_el):
                inj_pos = inj
                inj_neg = (inj + 1) % n_el
                for meas in range(n_el - 3):
                    meas_pos = (inj + 2 + meas) % n_el
                    meas_neg = (inj + 3 + meas) % n_el
                    rows.append([inj_pos, inj_neg, meas_pos, meas_neg])

        elif mode == "radial":
            step = n_el // 2
            for inj in range(n_el):
                inj_pos = inj
                inj_neg = (inj + step) % n_el
                for meas in range(n_el - 3):
                    meas_pos = (inj + 2 + meas) % n_el
                    meas_neg = (inj + 3 + meas) % n_el
                    if meas_pos in (inj_pos, inj_neg) or meas_neg in (inj_pos, inj_neg):
                        continue
                    rows.append([inj_pos, inj_neg, meas_pos, meas_neg])

        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        pattern = np.array(rows)  # shape (N, 4)

        np.savetxt(filepath, pattern, delimiter=",", fmt="%d")
        self.stimulation_pattern = pattern
        self.n_commands = len(pattern)
        self.log(f"Padrão {mode} gerado e salvo: {filepath} ({self.n_commands} comandos)")

    def run(self):
        self._running = True
        try:
            self.setup_serial()
            self.setup_eit()
            self.load_stimulation_pattern()

            if self._mode == "reference":
                self.log("Coletando referencia...")
                t0 = timer()
                raw = self.collect_measurements(self.config["avg"])
                if not self._running:
                    return
                self.log(f"Referencia coletada em {timer()-t0:.2f}s")
                self.v0 = self.normalize_array(self.remove_excitation_measurements(np.copy(raw)))
                ds = self.eit.solve(self.v0, self.v0, normalize=True)
                ds_n = sim2pts(self.pts, self.tri, np.real(ds))
                self.reference_done.emit(self.v0)
                self.frame_ready.emit(self.v0, np.zeros_like(self.v0), ds_n)
                self.log("Referencia concluida. Pronto para TDEIT.")
                self.finished_ok.emit()
                return

            if self._mode == "tdeit":
                # usa v0 ja coletado, nao recolleta
                v0_ext = self.config.get("v0", None)
                if v0_ext is None:
                    self.error_signal.emit("Referencia nao encontrada. Colete a referencia primeiro.")
                    return
                self.v0 = v0_ext
                self.log("Iniciando loop TDEIT com referencia existente...")
                while self._running:
                    t0 = timer()
                    raw = self.collect_measurements(self.config["avg"])
                    if not self._running:
                        break
                    self.log(f"Frame coletado em {timer()-t0:.2f}s")
                    v1 = self.normalize_array(self.remove_excitation_measurements(np.copy(raw)))
                    ds = self.eit.solve(v1, self.v0, normalize=True)
                    ds_n = sim2pts(self.pts, self.tri, np.real(ds))
                    self.frame_ready.emit(v1, v1 - self.v0, ds_n)

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.utom and self.utom.is_open:
                self.utom.close()
            self._running = False

    def stop(self): self._running = False
    def set_mode(self, mode): self._mode = mode

class EITPlaybackWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    reference_done = pyqtSignal(np.ndarray)
    log_message    = pyqtSignal(str)
    error_signal   = pyqtSignal(str)
    finished_ok    = pyqtSignal()

    def __init__(self, config: dict, frame_files: list, reference_file: str, delay_ms: int = 200, parent=None):
        super().__init__(parent)
        self.config = config
        self.frame_files = frame_files       # todos os frames (sem a referencia)
        self.reference_file = reference_file # arquivo escolhido como referencia
        self.delay_ms = delay_ms
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # inicia destravado
        self.pts = None
        self.tri = None
        self.x = None
        self.y = None
        self.mesh_obj = None
        self.eit = None
        self.v0 = None
        self._current_idx = 0
        self.frame_data = []  # Cache dos frames carregados

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_message.emit(f"[{ts}] {msg}")

    def advance(self):
        """Chamado pela GUI quando terminou de desenhar o frame atual."""
        pass  # Não usado mais com o novo sistema de pause

    def pause(self):
        """Pausa a reprodução."""
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        """Retoma a reprodução."""
        self._paused = False
        self._pause_event.set()

    def is_paused(self):
        return self._paused

    def toggle_pause(self):
        """Alterna entre pausado e reproduzindo."""
        if self._paused:
            self.resume()
        else:
            self.pause()

    def prev_frame(self):
        """Volta para o frame anterior."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._display_current_frame()

    def next_frame(self):
        """Avança para o próximo frame."""
        if self._current_idx < len(self.frame_data) - 1:
            self._current_idx += 1
            self._display_current_frame()

    def _display_current_frame(self):
        """Exibe o frame atual (usado na navegação manual)."""
        if 0 <= self._current_idx < len(self.frame_data):
            v1, diff, ds_n = self.frame_data[self._current_idx]
            self.frame_ready.emit(v1, diff, ds_n)

    def remove_excitation_measurements(self, data, window=3, threshold=2.0):
        data = data.astype(float)
        cleaned = data.copy()
        for i in range(len(data)):
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            neighbors = np.concatenate([data[start:i], data[i+1:end]])
            local_mean = np.mean(neighbors)
            local_std = np.std(neighbors)
            if abs(data[i] - local_mean) > threshold * local_std:
                cleaned[i] = local_mean
        return cleaned.astype(np.float32)

    def normalize_array(self, arr):
        arr_max = np.max(arr)
        return arr if arr_max == 0 else arr / arr_max

    def setup_eit(self):
        self.log("Configurando malha EIT...")
        n_el = self.config["n_el"]
        mode_idx = self.config.get("mode_idx", 0)
        dist_exc = 1 if mode_idx == 0 else n_el // 2
        
        self.mesh_obj = mesh.create(n_el, h0=self.config["h0"])
        self.pts = self.mesh_obj.node
        self.tri = self.mesh_obj.element
        self.x, self.y = self.pts[:, 0], self.pts[:, 1]
        protocol_obj = protocol.create(n_el, dist_exc=dist_exc, step_meas=1, parser_meas="std")
        EITForward(self.mesh_obj, protocol_obj).solve_eit()
        self.eit = jac.JAC(self.mesh_obj, protocol_obj)
        self.eit.setup(p=0.5, lamb=0.2, method="kotre", perm=1, jac_normalized=True)
        self.log(f"Malha EIT pronta ({n_el} eletrodos, modo {'adjacente' if mode_idx==0 else 'radial'}).")

    def _wait_gui(self):
        self._pause_event.wait()  
        self.msleep(self.delay_ms)

    def run(self):
        self._running = True
        self._paused = False
        self._pause_event.set()
        try:
            if not self.frame_files or not self.reference_file:
                self.error_signal.emit("Referência ou frames não selecionados.")
                return

            self.setup_eit()

            # Carrega e processa a referência
            raw0 = np.loadtxt(self.reference_file, dtype=np.float32)
            self.v0 = self.normalize_array(self.remove_excitation_measurements(np.copy(raw0)))
            ds = self.eit.solve(self.v0, self.v0, normalize=True)
            ds_n = sim2pts(self.pts, self.tri, np.real(ds))
            self.reference_done.emit(self.v0)
            self.frame_ready.emit(self.v0, np.zeros_like(self.v0), ds_n)
            self.log(f"Referencia: {os.path.basename(self.reference_file)}")
            self._wait_gui()

            # Pré-carrega e processa todos os frames
            self.log(f"Carregando {len(self.frame_files)} frames...")
            self.frame_data = []
            for idx, filepath in enumerate(self.frame_files, start=1):
                raw = np.loadtxt(filepath, dtype=np.float32)
                v1 = self.normalize_array(self.remove_excitation_measurements(np.copy(raw)))
                ds = self.eit.solve(v1, self.v0, normalize=True)
                ds_n = sim2pts(self.pts, self.tri, np.real(ds))
                diff = v1 - self.v0
                self.frame_data.append((v1, diff, ds_n))
                self.log(f"Frame {idx}/{len(self.frame_files)} carregado: {os.path.basename(filepath)}")

            # Reproduz os frames
            total = len(self.frame_data)
            self._current_idx = 0
            while self._running and self._current_idx < total:
                # Verifica se está pausado
                self._pause_event.wait()
                if not self._running:
                    break
                
                v1, diff, ds_n = self.frame_data[self._current_idx]
                self.frame_ready.emit(v1, diff, ds_n)
                self.log(f"Frame {self._current_idx + 1}/{total}")
                
                self._current_idx += 1
                
                # Espera o próximo frame
                self.msleep(self.delay_ms)

            self.log("Reproducao concluida. Use os botoes para navegar.")
            self.finished_ok.emit()
            self.pause()
            while self._running:
                self._pause_event.wait()
                if not self._running:
                    break
                self.msleep(100)

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self._running = False
        

    def stop(self):
        self._running = False
        self._pause_event.set()  # libera caso esteja esperando


# ══════════════════════════════════════════════════════════════════════════════
#  EXPLORADOR DE PASTAS
# ══════════════════════════════════════════════════════════════════════════════

class FolderExplorer(QWidget):
    path_changed = pyqtSignal(str)
    file_opened = pyqtSignal(str)  # NOVO: sinal para quando um arquivo é aberto

    def __init__(self, start_path: str = QDir.homePath(), parent=None):
        super().__init__(parent)
        self._back_stack    = []
        self._forward_stack = []
        self._current       = ""
        self._clipboard_path = None
        self._clipboard_cut  = False
        self._build_ui()
        self.navigate_to(start_path)
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        nav = QHBoxLayout()
        nav.setSpacing(3)

        btn_style = """
            QPushButton {
                background-color: #16213e; border: 1px solid #2a3a55;
                border-radius: 3px; color: #90b4d4;
                font-size: 14px; font-weight: bold;
                min-width: 26px; max-width: 26px;
                min-height: 26px; max-height: 26px;
            }
            QPushButton:hover   { background-color: #1e3a5f; color: #fff; }
            QPushButton:pressed { background-color: #00d4ff; color: #000; }
            QPushButton:disabled{ color: #2a3a55; border-color: #141e30; background-color: #0d1526; }
        """
        self.btn_back = QPushButton()
        self.btn_back.setIcon(qta.icon('fa5s.arrow-left', color='#90b4d4'))
        self.btn_back.setIconSize(QSize(18, 18))
        self.btn_back.setToolTip("Voltar  (Alt+Esquerda)")
        self.btn_back.setEnabled(False)
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setStyleSheet(btn_style)

        self.btn_fwd = QPushButton()
        self.btn_fwd.setIcon(qta.icon('fa5s.arrow-right', color='#90b4d4'))
        self.btn_fwd.setIconSize(QSize(18, 18))
        self.btn_fwd.setToolTip("Avançar  (Alt+Direita)")
        self.btn_fwd.setEnabled(False)
        self.btn_fwd.clicked.connect(self._go_forward)
        self.btn_fwd.setStyleSheet(btn_style)

        self.btn_up = QPushButton()
        self.btn_up.setIcon(qta.icon('fa5s.arrow-up', color='#90b4d4'))
        self.btn_up.setIconSize(QSize(18, 18))
        self.btn_up.setToolTip("Pasta acima  (Alt+Cima)")
        self.btn_up.clicked.connect(self._go_up)
        self.btn_up.setStyleSheet(btn_style)

        self.btn_refresh = QPushButton()
        self.btn_refresh.setIcon(qta.icon('fa5s.sync', color='#90b4d4'))
        self.btn_refresh.setIconSize(QSize(18, 18))
        self.btn_refresh.setToolTip("Atualizar (F5)")
        self.btn_refresh.clicked.connect(lambda: self.navigate_to(self._current, add_history=False))
        self.btn_refresh.setStyleSheet(btn_style)
        
        nav.addWidget(self.btn_back)
        nav.addWidget(self.btn_fwd)
        nav.addWidget(self.btn_up)
        nav.addWidget(self.btn_refresh)
        nav.addStretch()
        layout.addLayout(nav)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Caminho da pasta...")
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #0d0d1f; border: 1px solid #2a3a55;
                border-radius: 3px; color: #c8d8e8;
                font-size: 10px; padding: 3px 5px;
            }
            QLineEdit:focus { border-color: #00d4ff; }
        """)
        self.path_edit.returnPressed.connect(self._path_entered)
        layout.addWidget(self.path_edit)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Nome", "Tipo", "Modificado em"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #0d0d1f; border: 1px solid #1e2a40;
                border-radius: 3px; color: #c8c8d8; font-size: 11px; outline: none;
            }
            QTreeWidget::item { padding: 3px 2px; border-radius: 2px; }
            QTreeWidget::item:hover    { background-color: #1a2e4a; }
            QTreeWidget::item:selected { background-color: #1e4a7a; color: #fff; }
            QHeaderView::section {
                background-color: #111827; color: #6688aa; border: none;
                border-bottom: 1px solid #1e2a40; border-right: 1px solid #1e2a40;
                padding: 3px 4px; font-size: 10px;
            }
            QScrollBar:vertical { background: #0a0a1a; width: 8px; }
            QScrollBar::handle:vertical { background: #2a3a55; border-radius: 4px; }
        """)
        layout.addWidget(self.tree, stretch=1)

        bot = QHBoxLayout()
        bot.setSpacing(4)
        self.btn_new = QPushButton("+ Nova pasta")
        self.btn_new.setFixedHeight(24)
        self.btn_new.clicked.connect(self._new_folder)
        self.btn_new.setStyleSheet("""
            QPushButton {
                background-color: #131a2e; border: 1px solid #2a3a55;
                border-radius: 3px; color: #8899bb; font-size: 10px;
            }
            QPushButton:hover { background-color: #1e3a5f; color: #fff; }
        """)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #446688; font-size: 9px;")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bot.addWidget(self.btn_new)
        bot.addWidget(self.lbl_count, stretch=1)
        layout.addLayout(bot)

        self.tree.keyPressEvent = self._tree_key_press

    def navigate_to(self, path: str, add_history: bool = True):
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            return
        
        # bloqueia navegacao fora do workspace
        if hasattr(self, '_workspace_root') and self._workspace_root:
            if not path.startswith(self._workspace_root):
                return
        if add_history and self._current and self._current != path:
            self._back_stack.append(self._current)
            self._forward_stack.clear()
        self._current = path
        self.path_edit.setText(path)
        self.btn_back.setEnabled(len(self._back_stack) > 0)
        self.btn_fwd.setEnabled(len(self._forward_stack) > 0)
        self._refresh()
        self.path_changed.emit(path)
        
    def set_workspace_root(self, path: str):
        self._workspace_root = os.path.normpath(path)

    def _refresh(self):
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        try:
            entries = os.listdir(self._current)
        except PermissionError:
            entries = []

        folders, files = [], []
        for name in entries:
            full = os.path.join(self._current, name)
            (folders if os.path.isdir(full) else files).append(name)

        folders.sort(key=str.lower)
        files.sort(key=str.lower)

        for name in folders:
            full  = os.path.join(self._current, name)
            mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%d/%m/%Y %H:%M")
            item  = QTreeWidgetItem([name, "Pasta", mtime])
            item.setData(0, Qt.ItemDataRole.UserRole, full)
            item.setForeground(0, QColor("#7aadcc"))
            item.setForeground(1, QColor("#7a9cc7"))
            item.setForeground(2, QColor("#556677"))
            self.tree.addTopLevelItem(item)

        for name in files:
            full = os.path.join(self._current, name)
            ext  = os.path.splitext(name)[1].upper().lstrip(".") or "-"
            mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%d/%m/%Y %H:%M")
            item = QTreeWidgetItem([name, ext, mtime])
            item.setData(0, Qt.ItemDataRole.UserRole, full)
            item.setForeground(0, QColor("#c8c8d8"))
            item.setForeground(1, QColor("#557799"))
            item.setForeground(2, QColor("#556677"))
            self.tree.addTopLevelItem(item)

        self.tree.setSortingEnabled(True)
        self.lbl_count.setText(f"{len(folders)} pasta(s)  {len(files)} arquivo(s)")

    def _browse_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta", self._current)
        if path:
            self.navigate_to(path)
    
    def _on_double_click(self, item, col):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            if os.path.isdir(path):
                self.navigate_to(path)
            else:
                # Se for arquivo, emite sinal para abrir
                self.file_opened.emit(path)

    def _open_selected_file(self):
        """Abre o arquivo selecionado."""
        item = self.tree.currentItem()
        if not item:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            self.file_opened.emit(path)

    def _go_back(self):
        if self._back_stack:
            self._forward_stack.append(self._current)
            prev = self._back_stack.pop()
            self._current = ""
            self.navigate_to(prev, add_history=False)
            self.btn_fwd.setEnabled(True)

    def _go_forward(self):
        if self._forward_stack:
            self._back_stack.append(self._current)
            nxt = self._forward_stack.pop()
            self._current = ""
            self.navigate_to(nxt, add_history=False)

    def _go_up(self):
        parent = os.path.dirname(self._current)
        if parent and parent != self._current:
            self.navigate_to(parent)

    def _path_entered(self):
        self.navigate_to(self.path_edit.text().strip())

    def _tree_key_press(self, event):
        key  = event.key()
        mods = event.modifiers()
        item = self.tree.currentItem()
        path = item.data(0, Qt.ItemDataRole.UserRole) if item else None

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if path:
                if os.path.isdir(path):
                    self.navigate_to(path)
                else:
                    self.file_opened.emit(path)
        elif key == Qt.Key.Key_Delete:
            if path: self._delete_item(path)
        elif key == Qt.Key.Key_F2:
            if path: self._rename_item(path)
        elif key == Qt.Key.Key_F5:
            self.navigate_to(self._current, add_history=False)
        elif mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_C:
            if path:
                self._clipboard_path = path; self._clipboard_cut = False
                QApplication.clipboard().setText(path)
        elif mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_X:
            if path:
                self._clipboard_path = path; self._clipboard_cut = True
        elif mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_V:
            self._paste()
        elif key == Qt.Key.Key_Backspace:
            self._go_back() if self._back_stack else self._go_up()
        elif mods == Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_Left:
            self._go_back()
        elif mods == Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_Right:
            self._go_forward()
        elif mods == Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_Up:
            self._go_up()
        elif mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_O:
            self._browse_folder()
        else:
            QTreeWidget.keyPressEvent(self.tree, event)

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        path = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #16213e; border: 1px solid #334155; color: #e0e0e0; font-size: 11px; }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #2a5298; }
            QMenu::separator { height: 1px; background: #334155; margin: 3px 0; }
        """)
        
        act_open = None
        act_new     = menu.addAction("Nova pasta")
        act_refresh = menu.addAction("Atualizar  (F5)")
        menu.addSeparator()
        
        act_copy = act_cut = act_rename = act_delete = act_paste = act_copy_path = None
        if path:
            if os.path.isfile(path):
                act_open = menu.addAction("Abrir  (Enter)")
                menu.addSeparator()
            act_copy   = menu.addAction("Copiar  (Ctrl+C)")
            act_cut    = menu.addAction("Recortar  (Ctrl+X)")
            menu.addSeparator()
            act_rename = menu.addAction("Renomear  (F2)")
            act_delete = menu.addAction("Excluir  (Del)")
            menu.addSeparator()
            act_copy_path = menu.addAction("Copiar caminho")
        if self._clipboard_path:
            act_paste = menu.addAction("Colar  (Ctrl+V)")

        chosen = menu.exec(QCursor.pos())
        if not chosen: return
        
        if chosen == act_open and path:
            self.file_opened.emit(path)
        elif chosen == act_new:           self._new_folder()
        elif chosen == act_refresh:       self.navigate_to(self._current, add_history=False)
        elif chosen == act_rename and path: self._rename_item(path)
        elif chosen == act_delete and path: self._delete_item(path)
        elif chosen == act_copy   and path:
            self._clipboard_path = path; self._clipboard_cut = False
            QApplication.clipboard().setText(path)
        elif chosen == act_cut    and path:
            self._clipboard_path = path; self._clipboard_cut = True
        elif chosen == act_paste:       self._paste()
        elif chosen == act_copy_path and path: QApplication.clipboard().setText(path)

    def _new_folder(self):
        name, ok = QInputDialog.getText(self, "Nova pasta", "Nome da pasta:")
        if ok and name.strip():
            try:
                os.makedirs(os.path.join(self._current, name.strip()), exist_ok=True)
                self.navigate_to(self._current, add_history=False)
            except Exception as e:
                QMessageBox.warning(self, "Erro", str(e))

    def _rename_item(self, path):
        old_name = os.path.basename(path)
        new_name, ok = QInputDialog.getText(self, "Renomear", "Novo nome:", text=old_name)
        if ok and new_name.strip() and new_name != old_name:
            try:
                os.rename(path, os.path.join(os.path.dirname(path), new_name.strip()))
                self.navigate_to(self._current, add_history=False)
            except Exception as e:
                QMessageBox.warning(self, "Erro", str(e))

    def _delete_item(self, path):
        name = os.path.basename(path)
        resp = QMessageBox.question(self, "Confirmar exclusao", f"Excluir '{name}'?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
                self.navigate_to(self._current, add_history=False)
            except Exception as e:
                QMessageBox.warning(self, "Erro", str(e))

    def _paste(self):
        if not self._clipboard_path or not os.path.exists(self._clipboard_path):
            return
        name = os.path.basename(self._clipboard_path)
        dest = os.path.join(self._current, name)
        try:
            if self._clipboard_cut:
                shutil.move(self._clipboard_path, dest)
                self._clipboard_path = None; self._clipboard_cut = False
            else:
                if os.path.isdir(self._clipboard_path):
                    shutil.copytree(self._clipboard_path, dest)
                else:
                    shutil.copy2(self._clipboard_path, dest)
            self.navigate_to(self._current, add_history=False)
        except Exception as e:
            QMessageBox.warning(self, "Erro ao colar", str(e))

    def current_path(self): 
        return self._current

# ══════════════════════════════════════════════════════════════════════════════
#  JANELA DE CONFIGURACOES
# ══════════════════════════════════════════════════════════════════════════════

class ConfigDialog(QDialog):
    def __init__(self, current: dict, workspace_path: str = "", change_workspace_callback=None, parent=None):
        super().__init__(parent)
        self.workspace_path = workspace_path
        self.change_workspace_callback = change_workspace_callback
        self.setWindowTitle("Configurações")
        self.setMinimumWidth(440)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Serial ──
        grp_serial = QGroupBox("Comunicacao Serial")
        g = QGridLayout(grp_serial)
        g.setSpacing(8)

        g.addWidget(QLabel("Porta:"), 0, 0)
        port_row = QHBoxLayout()
        self.inp_port = QComboBox()
        self.inp_port.setEditable(True)
        self.inp_port.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._populate_ports(current.get("port", ""))
        btn_scan = QPushButton("Atualizar")
        btn_scan.setFixedHeight(28)
        btn_scan.setToolTip("Buscar portas COM disponíveis")
        btn_scan.clicked.connect(lambda: self._populate_ports(
            self.inp_port.currentData() or self.inp_port.currentText().split()[0]))
        port_row.addWidget(self.inp_port, stretch=1)
        port_row.addWidget(btn_scan)
        g.addLayout(port_row, 0, 1)

        g.addWidget(QLabel("Baudrate:"), 1, 0)
        self.inp_baud = QComboBox()
        self.inp_baud.addItems(["9600", "19200", "57600", "115200", "230400"])
        self.inp_baud.setCurrentText(str(current.get("baudrate", "115200")))
        g.addWidget(self.inp_baud, 1, 1)

        layout.addWidget(grp_serial)

        # ── EIT ──
        grp_eit = QGroupBox("Parametros EIT")
        e = QGridLayout(grp_eit)
        e.setSpacing(8)

        e.addWidget(QLabel("Selecione o Workspace:"), 0, 0, 1, 2)
        row_utom = QHBoxLayout()
        self.inp_path = QLineEdit()
        self.inp_path.setText(self.workspace_path)          
        self.inp_path.setReadOnly(True)                
        btn_browse_utom = QPushButton("Procurar")
        btn_browse_utom.setFixedHeight(28)
        btn_browse_utom.clicked.connect(self._browse_workspace)
        row_utom.addWidget(self.inp_path)
        row_utom.addWidget(btn_browse_utom)
        e.addLayout(row_utom, 1, 0, 1, 2)

        e.addWidget(QLabel("N eletrodos:"), 4, 0)
        self.inp_nel = QComboBox()
        self.inp_nel.addItems(["16", "32"])
        idx = 0 if current.get("n_el", 32) == 16 else 1
        self.inp_nel.setCurrentIndex(idx)
        e.addWidget(self.inp_nel, 4, 1)

        e.addWidget(QLabel("Resolucao malha (h0):"), 5, 0)
        self.inp_h0 = QDoubleSpinBox()
        self.inp_h0.setRange(0.01, 1.0)
        self.inp_h0.setSingleStep(0.01)
        self.inp_h0.setValue(current.get("h0", 0.1))
        e.addWidget(self.inp_h0, 5, 1)

        e.addWidget(QLabel("Frequencia:"), 6, 0)
        self.inp_freq = QComboBox()
        self.inp_freq.addItems(["10 kHz", "100 kHz"])
        self.inp_freq.setCurrentIndex(current.get("freq_idx", 0))
        e.addWidget(self.inp_freq, 6, 1)

        e.addWidget(QLabel("Gain (dB):"), 7, 0)
        self.inp_gain_db = QSpinBox()
        self.inp_gain_db.setRange(0, 120)
        self.inp_gain_db.setValue(current.get("gainDB", 60))
        e.addWidget(self.inp_gain_db, 7, 1)

        e.addWidget(QLabel("Gain linear:"), 8, 0)
        self.inp_gain_lin = QSpinBox()
        self.inp_gain_lin.setRange(0, 255)
        self.inp_gain_lin.setValue(current.get("gainLIN", 80))
        e.addWidget(self.inp_gain_lin, 8, 1)

        e.addWidget(QLabel("Medias:"), 9, 0)
        self.inp_avg = QSpinBox()
        self.inp_avg.setRange(1, 16)
        self.inp_avg.setValue(current.get("avg", 2))
        e.addWidget(self.inp_avg, 9, 1)

        # ── Modo de injeção ──
        e.addWidget(QLabel("Modo de injeção:"), 10, 0)
        self.inp_mode = QComboBox()
        self.inp_mode.addItems(["Adjacente", "Radial"])
        self.inp_mode.setCurrentIndex(current.get("mode_idx", 0))
        e.addWidget(self.inp_mode, 10, 1)

        layout.addWidget(grp_eit)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _populate_ports(self, current_port: str = ""):
        try:
            from serial.tools import list_ports
            ports = list_ports.comports()
        except Exception:
            ports = []
        self.inp_port.clear()
        selected_idx = 0
        for i, p in enumerate(sorted(ports, key=lambda x: x.device)):
            desc = p.description or ""
            label = f"{p.device}  -  {desc}" if p.device not in desc else p.device
            self.inp_port.addItem(label, userData=p.device)
            if p.device == current_port:
                selected_idx = i
        if ports:
            self.inp_port.setCurrentIndex(selected_idx)
        else:  
            if current_port:
                self.inp_port.addItem(current_port)
            else:
                self.inp_port.addItem("")
            self.inp_port.setCurrentIndex(0)

    def _browse_utom(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta com SW12.txt")
        if path:
            self.inp_path.setText(path)

    def _browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta de saida")
        if path:
            self.inp_data.setText(path)
            
    def _browse_workspace(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar workspace")
        if not folder:
            return
        
        folder = os.path.normpath(folder)
        home = os.path.normpath(os.path.expanduser("~"))
        
        bloqueadas = [home] + [os.path.normpath(os.path.join(home, p)) 
                    for p in ["Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"]]
        bloqueadas += [os.path.normpath(p) for p in ["C:\\", "C:\\Users", "C:\\Program Files", "C:\\Program Files (x86)"]]
        
        if folder in bloqueadas:
            QMessageBox.warning(self, "Pasta invalida",
                "Nao use pastas do sistema. Crie uma pasta nova para o workspace.")
            return
        
        conteudo = [f for f in os.listdir(folder) if f != "eit_workspace.json"]
        tem_workspace = os.path.exists(os.path.join(folder, "eit_workspace.json"))
        
        if not tem_workspace and len(conteudo) > 0:
            QMessageBox.warning(self, "Pasta invalida",
                "Selecione uma pasta vazia ou uma pasta que ja seja um workspace EIT.")
            return
        
        self.workspace_path = folder
        self.inp_path.setText(folder)
                
    def get_values(self) -> dict:
        port_data = self.inp_port.currentData()
        port = port_data if port_data else self.inp_port.currentText().split()[0].strip()
        return {
            "port": port,
            "baudrate": self.inp_baud.currentText(),
            "workspace": self.workspace_path,
            "n_el": int(self.inp_nel.currentText()),
            "h0": self.inp_h0.value(),
            "freq_idx": self.inp_freq.currentIndex(),
            "gainDB": self.inp_gain_db.value(),
            "gainLIN": self.inp_gain_lin.value(),
            "avg": self.inp_avg.value(),
            "mode_idx": self.inp_mode.currentIndex(),   # <-- ESSA LINHA
    }



class PlaybackConfigDialog(QDialog):
    def __init__(self, current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração da Aquisição")
        self.setMinimumWidth(320)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QGridLayout()
        form.setSpacing(8)
        form.addWidget(QLabel("N eletrodos:"), 0, 0)
        self.inp_nel = QComboBox()
        self.inp_nel.addItems(["16", "32"])
        self.inp_nel.setCurrentIndex(1 if current.get("n_el", 32) == 32 else 0)
        form.addWidget(self.inp_nel, 0, 1)
        form.addWidget(QLabel("Resolução malha (h0):"), 1, 0)
        self.inp_h0 = QDoubleSpinBox()
        self.inp_h0.setRange(0.01, 1.0)
        self.inp_h0.setSingleStep(0.01)
        self.inp_h0.setValue(current.get("h0", 0.1))
        form.addWidget(self.inp_h0, 1, 1)
        form.addWidget(QLabel("Modo de injeção:"), 2, 0)
        self.inp_mode = QComboBox()
        self.inp_mode.addItems(["Adjacente", "Radial"])
        self.inp_mode.setCurrentIndex(current.get("mode_idx", 0))
        form.addWidget(self.inp_mode, 2, 1)
        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    def get_values(self):
        return {
            "n_el": int(self.inp_nel.currentText()),
            "h0": self.inp_h0.value(),
            "mode_idx": self.inp_mode.currentIndex(),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  PAINEL DE PLOT 
# ══════════════════════════════════════════════════════════════════════════════

class PlotDock(QDockWidget):
    visibility_changed = pyqtSignal(str, bool)

    def __init__(self, panel_id: str, title: str, parent=None):
        super().__init__(title, parent)
        self.panel_id = panel_id
        self._bg = None   

        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setStyleSheet("""
            QDockWidget {
                color: #7aadcc; font-size: 10px; font-weight: bold;
                titlebar-close-icon: none;
            }
            QDockWidget::title {
                background-color: #111827; padding: 5px 8px;
                border-bottom: 1px solid #1e2a40;
            }
            QDockWidget::close-button, QDockWidget::float-button {
                background-color: #1e2a40; border: none; border-radius: 2px; padding: 2px;
            }
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {
                background-color: #2a5298;
            }
        """)

        inner = QWidget()
        inner.setStyleSheet("background-color: #0f0f23;")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        self.fig = Figure(facecolor="#1a1a2e")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        layout.addWidget(self.canvas)

        self.setWidget(inner)
        self.visibilityChanged.connect(lambda v: self.visibility_changed.emit(self.panel_id, v))
        
        self.canvas.mpl_connect('draw_event', self._on_draw)

    def _style_ax(self):
        self.ax.set_facecolor("#16213e")
        self.ax.tick_params(colors="#e0e0e0", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#334155")
        self.ax.grid(True, color="#334155", linewidth=0.5)
        self.fig.patch.set_facecolor("#1a1a2e")
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.12)

    def _on_draw(self, event):
        self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def capture_background(self):
        """Força um draw completo e guarda o fundo — chamar após configurar os artistas animados."""
        self.canvas.draw()
        self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def blit_artists(self, artists):
        """Restaura o fundo e desenha só os artistas passados (blitting)."""
        if self._bg is None:
            self.capture_background()
        self.canvas.restore_region(self._bg)
        for art in artists:
            self.ax.draw_artist(art)
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()

# ══════════════════════════════════════════════════════════════════════════════
#  GERENCIADOR DOS PAINEIS DE PLOT
# ══════════════════════════════════════════════════════════════════════════════

class EITPlotManager:
    PANELS = [
        ("eit",  "Reconstrução EIT"),
        ("v1",   "Dados medidos (v1)"),
        ("diff", "Diferença (v1 - v0)"),
    ]

    def __init__(self, main_window: QMainWindow):
        self.win = main_window
        self._docks: dict = {}
        self._tripcolor = None
        self._cbar = None
        self._electrodes_annotated = False
        self._full_redraw_counter = 0

        for pid, title in self.PANELS:
            dock = PlotDock(pid, title, main_window)
            self._docks[pid] = dock

        main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._docks["eit"])
        main_window.splitDockWidget(self._docks["eit"], self._docks["v1"], Qt.Orientation.Horizontal)
        main_window.splitDockWidget(self._docks["v1"], self._docks["diff"], Qt.Orientation.Vertical)
        main_window.resizeDocks([self._docks["eit"]], [600], Qt.Orientation.Horizontal)
        main_window.resizeDocks([self._docks["v1"], self._docks["diff"]], [300, 300], Qt.Orientation.Vertical)
        
        self._playback_bar = QWidget()
        self._playback_bar.setFixedHeight(32)
        self._playback_bar.setStyleSheet("""
            QWidget {
                background-color: #0d1526;
                border-bottom: 1px solid #1e2a40;
            }
            QToolTip {
                color: #e0e0e0;
                background-color: #1a1a3e;
                border: 1px solid #3a3a6a;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 10px;
            }
        """)
        bar_lay = QHBoxLayout(self._playback_bar)
        bar_lay.setContentsMargins(8, 2, 8, 2)
        bar_lay.setSpacing(4)

        btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 3px;
                min-width: 24px; max-width: 24px;
                min-height: 24px; max-height: 24px;
            }
            QPushButton:hover   { background-color: #1e3a5f; }
            QPushButton:pressed { background-color: #00d4ff; }
            QPushButton:disabled{ opacity: 0.3; }
            QPushButton::tooltip {
                color: #e0e0e0;
                background-color: #1a1a3e;
                border: 1px solid #3a3a6a;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 10px;
            }
        """

        self.btn_pb_prev = QPushButton()
        self.btn_pb_prev.setIcon(qta.icon('fa5s.step-backward', color='#6688aa'))
        self.btn_pb_prev.setIconSize(QSize(12, 12))
        self.btn_pb_prev.setStyleSheet(btn_style)
        self.btn_pb_prev.setToolTip("Frame anterior")

        self.btn_pb_pause = QPushButton()
        self.btn_pb_pause.setIcon(qta.icon('fa5s.pause', color='#6688aa'))
        self.btn_pb_pause.setIconSize(QSize(12, 12))
        self.btn_pb_pause.setStyleSheet(btn_style)
        self.btn_pb_pause.setToolTip("Pausar / Continuar")

        self.btn_pb_next = QPushButton()
        self.btn_pb_next.setIcon(qta.icon('fa5s.step-forward', color='#6688aa'))
        self.btn_pb_next.setIconSize(QSize(12, 12))
        self.btn_pb_next.setStyleSheet(btn_style)
        self.btn_pb_next.setToolTip("Próximo frame")

        # separador visual
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #1e2a40; margin: 4px 2px;")

        self.spin_pb_frame = QSpinBox()
        self.spin_pb_frame.setMinimum(1)
        self.spin_pb_frame.setMaximum(1)
        self.spin_pb_frame.setFixedSize(52, 22)
        self.spin_pb_frame.setStyleSheet("""
            QSpinBox {
                background-color: #16213e;
                border: 1px solid #2a3a55;
                border-radius: 3px;
                color: #c8d8e8;
                font-size: 10px;
                padding: 1px 3px;
            }
            QSpinBox::up-button, QSpinBox::down-button { width: 0px; }
        """)

        self.lbl_pb_total = QLabel("/ 0")
        self.lbl_pb_total.setStyleSheet("color: #445566; font-size: 10px;")

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #1e2a40; margin: 4px 2px;")

        self.btn_pb_save = QPushButton()
        self.btn_pb_save.setIcon(qta.icon('fa5s.file-pdf', color='#6688aa'))
        self.btn_pb_save.setIconSize(QSize(12, 12))
        self.btn_pb_save.setStyleSheet(btn_style)
        self.btn_pb_save.setToolTip("Salvar frame como PDF")

        bar_lay.addWidget(self.btn_pb_prev)
        bar_lay.addWidget(self.btn_pb_pause)
        bar_lay.addWidget(self.btn_pb_next)
        bar_lay.addWidget(sep)
        bar_lay.addWidget(self.spin_pb_frame)
        bar_lay.addWidget(self.lbl_pb_total)
        bar_lay.addStretch()
        bar_lay.addWidget(sep2)
        bar_lay.addWidget(self.btn_pb_save)

        self._playback_bar.setVisible(False)

        inner_layout = self._docks["eit"].widget().layout()
        inner_layout.setSpacing(0)
        inner_layout.insertWidget(0, self._playback_bar)

        ax3 = self._docks["eit"].ax
        ax3.set_title("Reconstrução EIT", color="#e0e0e0", fontsize=10)
        ax3.set_aspect("equal")
        ax3.set_xlim(-1.1, 1.1)
        ax3.set_ylim(-1.1, 1.1)
        self._docks["eit"].fig.tight_layout()
        self._docks["eit"].canvas.draw()
        ax3.grid(False)
        ax3.set_facecolor("#0a0a1a")

        ax1 = self._docks["v1"].ax
        ax1.set_title("Dados medidos (v1)", color="#e0e0e0", fontsize=9)
        ax1.set_xlabel("Medição", color="#e0e0e0", fontsize=8)
        ax1.set_ylabel("Magnitude", color="#e0e0e0", fontsize=8)
        self._line_v1, = ax1.plot([], [], color="#00d4ff", linewidth=1.2, animated=True)

        ax2 = self._docks["diff"].ax
        ax2.set_title("Diferença (v1 - v0)", color="#e0e0e0", fontsize=9)
        ax2.set_xlabel("Medição", color="#e0e0e0", fontsize=8)
        ax2.set_ylabel("Magnitude", color="#e0e0e0", fontsize=8)
        self._line_diff, = ax2.plot([], [], color="#ff6b6b", linewidth=1.2, animated=True)

    def dock(self, panel_id: str) -> PlotDock:
        return self._docks[panel_id]

    def set_panel_visible(self, panel_id: str, visible: bool):
        d = self._docks.get(panel_id)
        if not d:
            return
        d.setVisible(visible)
        if visible:
            d.raise_()

    def is_panel_visible(self, panel_id: str) -> bool:
        d = self._docks.get(panel_id)
        return d.isVisible() if d else False

    def redock_all(self):
        for d in self._docks.values():
            d.setFloating(False)
        for d in self._docks.values():
            self.win.removeDockWidget(d)
        self.win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._docks["eit"])
        self.win.splitDockWidget(self._docks["eit"], self._docks["v1"], Qt.Orientation.Horizontal)
        self.win.splitDockWidget(self._docks["v1"], self._docks["diff"], Qt.Orientation.Vertical)
        for d in self._docks.values():
            d.show()
        self.win.resizeDocks([self._docks["eit"]], [600], Qt.Orientation.Horizontal)
        self.win.resizeDocks([self._docks["v1"], self._docks["diff"]], [300, 300], Qt.Orientation.Vertical)
        # fundo mudou (docks realocados) — força recaptura no próximo update
        for d in self._docks.values():
            d._bg = None

    def update_plots(self, v1, diff, ds_n, pts, tri, x, y, mesh_obj):
        self._full_redraw_counter += 1
        do_full = (self._full_redraw_counter % 15 == 0)

        if self._docks["v1"].isVisible():
            self._line_v1.set_data(np.arange(len(v1)), v1)
            ax = self._docks["v1"].ax
            if do_full or self._docks["v1"]._bg is None:
                ax.relim(); ax.autoscale_view(); ax.margins(0.05)
                self._docks["v1"].capture_background()
            self._docks["v1"].blit_artists([self._line_v1])

        if self._docks["diff"].isVisible():
            self._line_diff.set_data(np.arange(len(diff)), diff)
            ax = self._docks["diff"].ax
            if do_full or self._docks["diff"]._bg is None:
                ax.relim(); ax.autoscale_view(); ax.margins(0.05)
                self._docks["diff"].capture_background()
            self._docks["diff"].blit_artists([self._line_diff])

        if self._docks["eit"].isVisible():
            ax = self._docks["eit"].ax
            dock = self._docks["eit"]

            if self._tripcolor is None:
                self._tripcolor = ax.tripcolor(
                    x, y, tri, ds_n, 
                    shading="gouraud", 
                    cmap="RdBu_r", 
                    alpha=0.95,
                    animated=True,
                    zorder=1 
                )
                fig = dock.fig
                self._cbar = fig.colorbar(self._tripcolor, ax=ax, fraction=0.046, shrink=0.7)
                self._cbar.ax.yaxis.set_tick_params(color="#e0e0e0", labelsize=7)
                self._cbar.set_label("Condutividade (Δσ)", color="#e0e0e0", fontsize=8)
                
                # Cria os eletrodos quando o tripcolor é criado pela primeira vez
                if hasattr(mesh_obj, 'el_pos'):
                    el_x = [x[e] for e in mesh_obj.el_pos]
                    el_y = [y[e] for e in mesh_obj.el_pos]
                    
                    # Guarda referência do scatter
                    self._electrode_scatters = ax.scatter(
                        el_x, el_y, 
                        color='white', 
                        s=40, 
                        zorder=5, 
                        edgecolors='#ffd700', 
                        linewidth=1.5,
                        animated=True
                    )
                    
                    # Guarda referências dos textos
                    self._electrode_texts = []
                    for i, e in enumerate(mesh_obj.el_pos):
                        t = ax.annotate(
                            str(i + 1), 
                            xy=(x[e], y[e]),
                            color="#ffd700", 
                            fontsize=9,
                            fontweight="bold",
                            ha="center", 
                            va="center",
                            zorder=10,
                            animated=True,
                            bbox=dict(
                                boxstyle="circle,pad=0.15", 
                                facecolor="#0a0a1a", 
                                alpha=0.8,
                                edgecolor="none"
                            )
                        )
                        self._electrode_texts.append(t)
                    self._electrodes_annotated = True
            else:
                self._tripcolor.set_array(ds_n)

            vmin, vmax = float(np.min(ds_n)), float(np.max(ds_n))
            self._tripcolor.set_clim(vmin, vmax)

            if do_full or dock._bg is None:
                self._cbar.update_normal(self._tripcolor)
                dock.capture_background()
            
            # Blit todos os artistas juntos: tripcolor + eletrodos
            artists_to_blit = [self._tripcolor]
            if hasattr(self, '_electrode_scatters') and self._electrode_scatters is not None:
                artists_to_blit.append(self._electrode_scatters)
            if hasattr(self, '_electrode_texts') and self._electrode_texts:
                artists_to_blit.extend(self._electrode_texts)
                
            dock.blit_artists(artists_to_blit)

# ══════════════════════════════════════════════════════════════════════════════
#  JANELA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class EITMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIT Tomographer")
        self.resize(1400, 800)
        self._apply_dark_theme()
        self.worker = None
        self.playback_worker = None
        self._latest_frame = None
        self._prefs = load_prefs()
        self._config = {}
        self._config.update(self._prefs.get("config", {}))
        self._is_configuring = False
        self._build_menubar()
        self._build_ui()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(120)   
        self._render_timer.timeout.connect(self._render_latest_frame)
        self._render_timer.start()
    
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget, QDialog {
                background-color: #0f0f23; color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif; font-size: 12px;
            }
            QMenuBar { background-color: #0a0a1a; color: #e0e0e0; border-bottom: 1px solid #334155; }
            QMenuBar::item:selected { background-color: #1e3a5f; }
            QMenu { background-color: #16213e; border: 1px solid #334155; color: #e0e0e0; }
            QMenu::item:selected { background-color: #2a5298; }
            QMenu::separator { height: 1px; background: #334155; margin: 3px 0; }
            QGroupBox {
                border: 1px solid #1e2a40; border-radius: 4px;
                margin-top: 8px; padding-top: 6px;
                font-weight: bold; color: #4488bb; font-size: 11px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #16213e; border: 1px solid #334155;
                border-radius: 4px; padding: 4px 6px; color: #e0e0e0;
            }
            QLineEdit:focus, QComboBox:focus { border-color: #00d4ff; }
            QPushButton {
                background-color: #1e3a5f; border: 1px solid #334155;
                border-radius: 5px; padding: 6px 12px; color: #e0e0e0; font-weight: bold;
            }
            QPushButton:hover   { background-color: #2a5298; border-color: #00d4ff; }
            QPushButton:pressed { background-color: #00d4ff; color: #0f0f23; }
            QPushButton:disabled{ background-color: #1a1a2e; color: #555; }
            QTextEdit {
                background-color: #0a0a1a; border: 1px solid #1e2a40;
                border-radius: 3px; color: #aaffaa;
                font-family: 'Consolas', monospace; font-size: 10px;
            }
            QStatusBar { background-color: #0a0a1a; color: #aaa; font-size: 10px; }
            QSplitter::handle { background-color: #1e2a40; width: 2px; }
            QDialogButtonBox QPushButton { min-width: 80px; }
        """)
    
    def _generate_pattern_file(self, dest_folder, n_el, mode_idx):
        """Gera o arquivo de padrões de estimulação (928 linhas x 4 colunas)."""
        os.makedirs(dest_folder, exist_ok=True)
        
        rows = []
        
        if mode_idx == 0:  # Adjacente
            for inj in range(n_el):
                inj_pos = inj
                inj_neg = (inj + 1) % n_el
                for meas in range(n_el - 3):
                    meas_pos = (inj + 2 + meas) % n_el
                    meas_neg = (inj + 3 + meas) % n_el
                    rows.append([inj_pos, inj_neg, meas_pos, meas_neg])
        else:  # Radial
            for inj in range(n_el):
                inj_pos = inj
                inj_neg = (inj + n_el // 2) % n_el
                for meas in range(n_el - 3):
                    meas_pos = (inj + 2 + meas) % n_el
                    meas_neg = (inj + 3 + meas) % n_el
                    rows.append([inj_pos, inj_neg, meas_pos, meas_neg])
        
        pattern = np.array(rows)  # shape (N, 4)
        
        filename = f"stim_pattern_{n_el}_{mode_idx}.txt"
        filepath = os.path.join(dest_folder, filename)
        np.savetxt(filepath, pattern, delimiter=",", fmt="%d")
        
        modo_nome = "Adjacente" if mode_idx == 0 else "Radial"
        self.log(f"Padrão {modo_nome} gerado: {filepath} ({len(pattern)} comandos)")
        return filepath
    
    def _get_data_folder(self):
        """Retorna o caminho da pasta 'data' dentro do workspace atual, criando-a se não existir."""
        workspace = self.explorer.current_path()
        data_folder = os.path.join(workspace, "data")
        os.makedirs(data_folder, exist_ok=True)
        return data_folder

    def _get_eit_folder(self):
        workspace = self.explorer.current_path()
        if not workspace:
            return None
        eit_folder = os.path.join(workspace, "config")
        os.makedirs(eit_folder, exist_ok=True)
        return eit_folder

    def _build_menubar(self):
        mb = self.menuBar()

        m_file = mb.addMenu("Arquivo")
        act_save = QAction("Salvar frame atual", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_current)
        m_file.addAction(act_save)
        
        # Salvar como PDF
        act_save_pdf = QAction("Salvar imagem EIT como PDF...", self)
        act_save_pdf.setShortcut("Ctrl+Shift+S")
        act_save_pdf.triggered.connect(self._save_pdf)
        m_file.addAction(act_save_pdf)
        
        m_file.addSeparator()
        act_exit = QAction("Sair", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_cfg = mb.addMenu("Configurações")
        act_cfg = QAction("Abrir configurações...", self)
        act_cfg.setShortcut("Ctrl+P")
        act_cfg.triggered.connect(self._open_config)
        m_cfg.addAction(act_cfg)

        m_ctrl = mb.addMenu("Controle")
        self.act_ref = QAction("Coletar Referencia", self)
        self.act_ref.triggered.connect(self._start_reference)
        self.act_playback = QAction("Reproduzir frames salvos...", self)
        self.act_playback.triggered.connect(self._start_playback)
        m_ctrl.addAction(self.act_playback)
        m_ctrl.addAction(self.act_ref)
        self.act_tdeit = QAction("Iniciar TDEIT", self)
        self.act_tdeit.setEnabled(False)
        self.act_tdeit.triggered.connect(self._start_tdeit)
        m_ctrl.addAction(self.act_tdeit)
        
        # Adiciona ações de controle de playback
        m_ctrl.addSeparator()
        self.act_pause = QAction("Pausar/Continuar  (Space)", self)
        self.act_pause.setShortcut("Space")
        self.act_pause.setEnabled(False)
        self.act_pause.triggered.connect(self._toggle_pause)
        m_ctrl.addAction(self.act_pause)

        self.act_prev = QAction("Frame anterior  (Left)", self)
        self.act_prev.setShortcut("Left")
        self.act_prev.setEnabled(False)
        self.act_prev.triggered.connect(self._prev_frame)
        m_ctrl.addAction(self.act_prev)

        self.act_next = QAction("Proximo frame  (Right)", self)
        self.act_next.setShortcut("Right")
        self.act_next.setEnabled(False)
        self.act_next.triggered.connect(self._next_frame)
        m_ctrl.addAction(self.act_next)
        
        m_ctrl.addSeparator()
        self.act_stop = QAction("Parar", self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self._stop)
        m_ctrl.addAction(self.act_stop)

        # menu Visualizar — paineis e painel lateral
        m_view = mb.addMenu("Visualizar")
        self.act_toggle_panel = QAction("Ocultar painel lateral", self)
        self.act_toggle_panel.setShortcut("Ctrl+B")
        self.act_toggle_panel.triggered.connect(self._toggle_panel)
        m_view.addAction(self.act_toggle_panel)

        act_fs = QAction("Tela cheia  (F11)", self)
        act_fs.setShortcut("F11")
        act_fs.triggered.connect(self._toggle_fullscreen)
        m_view.addAction(act_fs)

        m_view.addSeparator()

        self._panel_actions = {}
        labels = {"eit": "Reconstrução EIT", "v1": "Dados medidos (v1)", "diff": "Diferença (v1 - v0)"}
        for i, (pid, lbl) in enumerate(labels.items(), start=1):
            act = QAction(f"Mostrar: {lbl}", self)
            act.setCheckable(True)
            act.setChecked(True)
            act.setShortcut(f"Ctrl+{i}")
            act.triggered.connect(lambda checked, p=pid: self._toggle_plot(p, checked))
            m_view.addAction(act)
            self._panel_actions[pid] = act

        m_view.addSeparator()
        act_redock = QAction("Reencaixar paineis na posicao padrao", self)
        act_redock.setShortcut("Ctrl+L")
        act_redock.triggered.connect(self._redock_plots)
        m_view.addAction(act_redock)

    def _build_ui(self):
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks
        )

        # dock do explorador + log 
        self.left_dock = QDockWidget("Workspace", self)
        self.left_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.left_dock.setStyleSheet("""
            QDockWidget { color: #4488bb; font-size: 11px; font-weight: bold; }
            QDockWidget::title {
                background-color: #111827;
                padding: 5px 8px;
                border-bottom: 1px solid #1e2a40;
            }
            QDockWidget::close-button, QDockWidget::float-button {
                background-color: #1e2a40; border: none; border-radius: 2px; padding: 2px;
            }
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {
                background-color: #2a5298;
            }
        """)

        left_inner = QWidget()
        left_inner.setMinimumWidth(290)
        left_inner.setMaximumWidth(350)
        
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        start = self._prefs.get("last_folder", os.path.expanduser("~"))
        if not os.path.isdir(start):
            start = os.path.expanduser("~")
        self.explorer = FolderExplorer(start_path=start)
        self.explorer.path_changed.connect(self._on_path_changed)
        self.explorer.file_opened.connect(self._open_file) 
        left_layout.addWidget(self.explorer, stretch=1)

        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(3, 10, 3, 3)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        log_lay.addWidget(self.log_box)
        left_layout.addWidget(log_grp)

        self.left_dock.setWidget(left_inner)
        self.left_dock.visibilityChanged.connect(self._on_left_dock_visibility)
        
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)

        # ── docks dos plots com novo layout ──
        self.plot_mgr = EITPlotManager(self)
        for pid in ("eit", "v1", "diff"):
            self.plot_mgr.dock(pid).visibility_changed.connect(self._on_plot_panel_visibility)

        # Os docks já são adicionados pelo EITPlotManager, não precisa adicionar novamente

        self._panel_visible = True
                        
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        if not DEPS_OK:
            self.log(f"Dependencia faltando: {MISSING_DEP}")
    
    def _save_pdf(self):
        """Salva a imagem EIT como PDF."""
        workspace = self._config.get("workspace", ".")
        img_folder = os.path.join(workspace, "imagens")
        os.makedirs(img_folder, exist_ok=True)
        
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_path = os.path.join(img_folder, f"eit_{ts}.pdf")
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar EIT como PDF", default_path, "PDF (*.pdf)")
        if not path:
            return
        
        fig = self.plot_mgr.dock("eit").fig
        fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
        self.log(f"Imagem salva: {path}")
        self.explorer.navigate_to(self.explorer.current_path(), add_history=False)  # atualiza explorer

    def _start_playback(self):
        workspace = self._config.get("workspace", "")
        default_folder = os.path.join(workspace, "data") if workspace else ""
        
        if default_folder and os.path.isdir(default_folder):
            resposta = QMessageBox.question(
                self, "Pasta de dados",
                f"Usar pasta de dados do workspace?\n{default_folder}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if resposta == QMessageBox.StandardButton.Yes:
                folder = default_folder
            else:
                folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta com frames")
        else:
            folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta com frames")
        
        if not folder:
            return

        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith(".txt")
        )
        if not files:
            QMessageBox.warning(self, "Nenhum arquivo", "Nenhum .txt encontrado nessa pasta.")
            return

        nomes = [os.path.basename(f) for f in files]
        escolha, ok = QInputDialog.getItem(
            self, "Selecionar referência", "Escolha o arquivo de referência (v0):",
            nomes, editable=False
        )
        if not ok:
            return

        reference_file = files[nomes.index(escolha)]
        frame_files = [f for f in files if f != reference_file]
        if not frame_files:
            QMessageBox.warning(self, "Sem frames", "Não há outros frames além da referência selecionada.")
            return

        dlg = PlaybackConfigDialog(self._config, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        playback_config = dlg.get_values()
        playback_config["workspace"] = self._config.get("workspace", "")

        self.playback_worker = EITPlaybackWorker(playback_config, frame_files, reference_file, delay_ms=150)
        self.playback_worker.log_message.connect(self.log)
        self.playback_worker.error_signal.connect(self._on_error)
        self.playback_worker.frame_ready.connect(self._on_frame_playback)
        self.playback_worker.reference_done.connect(self._on_reference_done)
        self.playback_worker.finished_ok.connect(self._on_worker_done)
        self.playback_worker.finished.connect(self._on_worker_finished)
        self.playback_worker.start()
        self._set_running(True)
        self.status.showMessage(
            f"Playback rodando (n_el={playback_config['n_el']}, "
            f"modo={'Adjacente' if playback_config['mode_idx']==0 else 'Radial'})..."
        )

        self.act_pause.setEnabled(True)
        self.act_prev.setEnabled(True)
        self.act_next.setEnabled(True)
        self.act_pause.setText("Pausar/Continuar  (Space)")
        
        total = len(frame_files)
        self.plot_mgr._playback_bar.setVisible(True)
        self.plot_mgr.spin_pb_frame.setMaximum(total)
        self.plot_mgr.spin_pb_frame.setValue(1)
        self.plot_mgr.lbl_pb_total.setText(f"/ {total}")

        self.plot_mgr.btn_pb_prev.clicked.connect(self._prev_frame)
        self.plot_mgr.btn_pb_next.clicked.connect(self._next_frame)
        self.plot_mgr.btn_pb_pause.clicked.connect(self._toggle_pause)
        self.plot_mgr.btn_pb_save.clicked.connect(self._save_pdf)
        self.plot_mgr.spin_pb_frame.valueChanged.connect(self._seek_frame)


    def _seek_frame(self, idx: int):
        if self.playback_worker and self.playback_worker.frame_data:
            self.playback_worker._current_idx = idx - 1 
            self.playback_worker._display_current_frame()
        
        
    def _open_file(self, filepath: str):
        """Abre um arquivo com o programa padrão do sistema."""
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(filepath)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', filepath])
            else:  # Linux
                subprocess.run(['xdg-open', filepath])
            
            self.log(f"Arquivo aberto: {os.path.basename(filepath)}")
            
        except Exception as e:
            self.log(f"Erro ao abrir arquivo: {str(e)}")
            QMessageBox.warning(self, "Erro ao abrir", 
                f"Não foi possível abrir o arquivo:\n{filepath}\n\nErro: {str(e)}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_panel(self):
        visible = self.left_dock.isVisible()
        self.left_dock.setVisible(not visible)

    def _on_left_dock_visibility(self, visible: bool):
        self._panel_visible = visible
        self.act_toggle_panel.setText("Mostrar painel lateral" if not visible else "Ocultar painel lateral")

    def _toggle_plot(self, panel_id: str, checked: bool):
        self.plot_mgr.set_panel_visible(panel_id, checked)

    def _on_plot_panel_visibility(self, panel_id: str, visible: bool):
        # sincroniza o checkmark no menu quando o usuario fecha pelo X do dock
        if panel_id in self._panel_actions:
            self._panel_actions[panel_id].blockSignals(True)
            self._panel_actions[panel_id].setChecked(visible)
            self._panel_actions[panel_id].blockSignals(False)

    def _redock_plots(self):
        self.plot_mgr.redock_all()

    def _on_path_changed(self, path: str):
        self._prefs["last_folder"] = path
        save_prefs(self._prefs)
    
    def _open_config(self):
        self._is_configuring = True
        dlg = ConfigDialog(
            self._config,
            workspace_path=self._config.get("workspace", ""),
            change_workspace_callback=None,
            parent=self
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            values = dlg.get_values()
            workspace = values.get("workspace", "")
            
            if not workspace or not os.path.isdir(workspace):
                QMessageBox.warning(self, "Workspace inválido", "Selecione uma pasta válida.")
                self._is_configuring = False
                self._open_config()
                return
            
            # Atualiza configurações
            self._config = values
            self._config["workspace"] = workspace
            self._prefs["last_workspace"] = workspace
            self._prefs["config"] = self._config
            save_prefs(self._prefs, workspace)
            
            # Navega para o workspace apenas se mudou de pasta
            workspace_atual = self.explorer.current_path()
            if workspace_atual != workspace:
                self.explorer.set_workspace_root(workspace)
                self.explorer.navigate_to(workspace, add_history=False)
            else:
                # Já está no workspace, só atualiza a raiz para bloquear navegação
                self.explorer.set_workspace_root(workspace)
            
            # Cria a pasta .eit e gera o arquivo de padrões dentro dela
            eit_folder = os.path.join(workspace, "config")
            os.makedirs(eit_folder, exist_ok=True)
            self._generate_pattern_file(eit_folder, self._config["n_el"], self._config["mode_idx"])
            
            self.log(f"Workspace configurado: {workspace}")
            self.log(f"Arquivo de padrões gerado/atualizado para {self._config['n_el']} eletrodos, modo {['Adjacente','Radial'][self._config['mode_idx']]}")
        else:
            if not self._config.get("workspace"):
                QMessageBox.critical(self, "Sem workspace", "Nenhum workspace selecionado. O programa será encerrado.")
                QApplication.quit()
        
        self._is_configuring = False
            
    def _load_workspace_config(self):
        if self._is_configuring:
            return

        prefs = load_prefs()
        last_workspace = prefs.get("last_workspace", "")

        if last_workspace and os.path.isdir(last_workspace):
            eit_folder = os.path.join(last_workspace, ".eit")
            ws_file = os.path.join(eit_folder, "eit_workspace.json")
            if os.path.exists(ws_file):
                try:
                    with open(ws_file, "r") as f:
                        ws_config = json.load(f)
                    self._config = ws_config.get("config", {})
                    self.explorer.set_workspace_root(last_workspace)
                    self.explorer.navigate_to(last_workspace, add_history=False)
                    self.log(f"Workspace carregado: {last_workspace}")

                    # Verifica se o arquivo de padrões existe, se não, gera
                    if self._config.get("n_el") and self._config.get("mode_idx") is not None:
                        pattern_file = os.path.join(eit_folder,
                            f"stim_pattern_{self._config['n_el']}_{self._config['mode_idx']}.txt")
                        if not os.path.exists(pattern_file):
                            self._generate_pattern_file(eit_folder, self._config["n_el"], self._config["mode_idx"])
                            self.log("Arquivo de padrões gerado automaticamente.")
                    return
                except Exception as e:
                    self.log(f"Erro ao carregar workspace: {e}")

        self._open_config()

    def _make_serial_config(self):
        if not self._config:
            self.log("ERRO: Nenhuma configuração salva. Abra Configurações e configure.")
            return None

        required = ["port", "baudrate", "workspace", "n_el", "h0", "freq_idx", "gainDB", "gainLIN", "avg", "mode_idx"]
        for key in required:
            if key not in self._config or self._config[key] in (None, ""):
                self.log(f"ERRO: '{key}' não configurado. Abra Configurações.")
                return None

        data_folder = self._get_data_folder()

        config = {
            "port": self._config["port"],
            "baudrate": int(self._config["baudrate"]),
            "data_bits": 8,
            "parity": serial.PARITY_ODD,
            "stop_bits": serial.STOPBITS_ONE,
            "workspace": self._config["workspace"],
            "data_folder": data_folder,
            "n_el": self._config["n_el"],
            "h0": self._config["h0"],
            "freq_sel": self._config["freq_idx"] + 1,
            "gainDB": self._config["gainDB"],
            "gainLIN": self._config["gainLIN"],
            "avg": self._config["avg"],
            "mode_idx": self._config["mode_idx"],
        }

        # Adiciona v0 se existir
        if hasattr(self, '_v0_reference') and self._v0_reference is not None:
            config["v0"] = self._v0_reference

        return config

    def _start_worker(self, mode):
        if not DEPS_OK:
            self.log(f"Dependencias faltando: {MISSING_DEP}")
            return
        if self.worker and self.worker.isRunning():
            return
        
        config = self._make_serial_config()
        if config is None:
            return
        
        self.worker = EITWorker(config)
        self.worker.set_mode(mode)
        self.worker.log_message.connect(self.log)
        self.worker.error_signal.connect(self._on_error)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.reference_done.connect(self._on_reference_done)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self._set_running(True)
        self.status.showMessage(f"Modo: {mode.upper()} - rodando...")

    def _start_reference(self): self._start_worker("reference")
    def _start_tdeit(self):     self._start_worker("tdeit")

    def _toggle_pause(self):
        """Alterna entre pausado e reproduzindo durante o playback."""
        if self.playback_worker and self.playback_worker.isRunning():
            if self.playback_worker.is_paused():
                self.playback_worker.resume()
                self.act_pause.setText("Pausar/Continuar  (Space)")
                self.status.showMessage("Playback continuando...")
                self.log("Playback continuando")
            else:
                self.playback_worker.pause()
                self.act_pause.setText("Continuar  (Space)")
                self.status.showMessage("Playback pausado")
                self.log("Playback pausado")

    def _prev_frame(self):
        """Volta para o frame anterior durante o playback."""
        if self.playback_worker and self.playback_worker.isRunning():
            self.playback_worker.prev_frame()
            self.log("Frame anterior")

    def _next_frame(self):
        """Avança para o próximo frame durante o playback."""
        if self.playback_worker and self.playback_worker.isRunning():
            self.playback_worker.next_frame()
            self.log("Próximo frame")

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if hasattr(self, 'playback_worker') and self.playback_worker and self.playback_worker.isRunning():
            self.playback_worker.stop()
        self.act_stop.setEnabled(False)
        self.act_pause.setEnabled(False)
        self.act_prev.setEnabled(False)
        self.act_next.setEnabled(False)
        self.status.showMessage("Parando...")

    def _set_running(self, running):
        self.act_ref.setEnabled(not running)
        self.act_stop.setEnabled(running)
        if running: 
            self.act_tdeit.setEnabled(False)
        else:
            # Quando parar, desabilita controles de playback
            if not self.playback_worker or not self.playback_worker.isRunning():
                self.act_pause.setEnabled(False)
                self.act_prev.setEnabled(False)
                self.act_next.setEnabled(False)
                self.act_pause.setText("Pausar/Continuar  (Space)")

    def _save_current(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar frame", "./data/frame_export.txt", "Texto (*.txt);;NumPy (*.npy)")
        if path:
            self.log(f"Frame salvo em {path}")

    def log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        self.status.showMessage(msg)

    def _on_frame(self, v1, diff, ds_n):
        if not self.worker: return
        w = self.worker
        self._latest_frame = (v1, diff, ds_n, w.pts, w.tri, w.x, w.y, w.mesh_obj)

    def _on_frame_playback(self, v1, diff, ds_n):
        w = self.playback_worker
        self._latest_frame = (v1, diff, ds_n, w.pts, w.tri, w.x, w.y, w.mesh_obj)
        self.plot_mgr.spin_pb_frame.blockSignals(True)
        self.plot_mgr.spin_pb_frame.setValue(w._current_idx + 1)
        self.plot_mgr.spin_pb_frame.blockSignals(False)
    
    def _render_latest_frame(self):
        if self._latest_frame is None:
            return
        v1, diff, ds_n, pts, tri, x, y, mesh_obj = self._latest_frame
        self.plot_mgr.update_plots(v1, diff, ds_n, pts, tri, x, y, mesh_obj)
    
    def _on_reference_done(self, v0: np.ndarray):
        self._v0_reference = v0.copy()
        self.log("Referencia capturada.")
        self.act_tdeit.setEnabled(True)

    def _on_worker_done(self):
        if self.playback_worker and self.playback_worker.isRunning():
            self.act_ref.setEnabled(True)
            self.act_stop.setEnabled(True)
            self.act_tdeit.setEnabled(False)
            self.status.showMessage("Reprodução concluida.")
        else:
            self.act_ref.setEnabled(True)
            self.act_stop.setEnabled(False)
            self.act_tdeit.setEnabled(True)
            self.status.showMessage("Referencia concluida.")
        
    def _on_worker_finished(self):
        self._set_running(False)
        self.plot_mgr._playback_bar.setVisible(False)
        self.status.showMessage("Parado.")
        self.act_pause.setEnabled(False)
        self.act_prev.setEnabled(False)
        self.act_next.setEnabled(False)
        self.act_pause.setText("Pausar/Continuar  (Space)")

    def _on_error(self, msg):
        self.log(f"ERRO: {msg}")
        self._set_running(False)
        self.act_tdeit.setEnabled(False)
        self.status.showMessage(f"Erro: {msg}")
        # Desabilita controles de playback
        self.act_pause.setEnabled(False)
        self.act_prev.setEnabled(False)
        self.act_next.setEnabled(False)
        self.act_pause.setText("Pausar/Continuar  (Space)")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = EITMainWindow()
    win.show()
    
    QTimer.singleShot(300, win._load_workspace_config)
    
    sys.exit(app.exec())