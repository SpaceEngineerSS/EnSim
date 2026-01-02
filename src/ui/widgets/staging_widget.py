"""
Multi-Stage Vehicle Configuration Widget.

Provides UI for configuring multi-stage rockets with:
- Stage addition/removal
- Engine parameters per stage
- Mass properties
- Delta-V calculations
- Pre-built vehicle templates (Falcon 9, Saturn V)
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QScrollArea, QMessageBox,
    QSplitter, QProgressBar
)
from PyQt6.QtGui import QFont

from ...core.staging import (
    Stage, StageEngine, MultiStageVehicle,
    StagingTrigger, StageStatus,
    create_falcon_9_like, create_saturn_v_like, create_custom_vehicle
)


class StageConfigCard(QFrame):
    """Configuration card for a single stage."""
    
    stage_changed = pyqtSignal()
    remove_requested = pyqtSignal(int)
    
    def __init__(self, stage_num: int, parent=None):
        super().__init__(parent)
        self.stage_num = stage_num
        self.setObjectName("stageCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(450)  # Prevent collapse in scroll area
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header with stage number and remove button
        header = QHBoxLayout()
        title = QLabel(f"Stage {self.stage_num}")
        title.setObjectName("summaryValue")  # Large, clean title
        header.addWidget(title)
        header.addStretch()
        
        self.remove_btn = QPushButton("×")
        self.remove_btn.setObjectName("removeStageBtn")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.stage_num))
        header.addWidget(self.remove_btn)
        layout.addLayout(header)
        
        # Mass properties
        mass_group = QGroupBox("Mass Properties")
        mass_layout = QFormLayout(mass_group)
        mass_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        mass_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self.dry_mass = QDoubleSpinBox()
        self.dry_mass.setRange(100, 500000)
        self.dry_mass.setValue(5000)
        self.dry_mass.setSuffix(" kg")
        self.dry_mass.setDecimals(0)
        self.dry_mass.valueChanged.connect(self._emit_change)
        
        mass_layout.addRow("Dry Mass:", self.dry_mass)
        
        self.prop_mass = QDoubleSpinBox()
        self.prop_mass.setRange(1000, 5000000)
        self.prop_mass.setValue(50000)
        self.prop_mass.setSuffix(" kg")
        self.prop_mass.setDecimals(0)
        self.prop_mass.valueChanged.connect(self._emit_change)
        mass_layout.addRow("Propellant:", self.prop_mass)
        
        layout.addWidget(mass_group)
        
        # Engine properties
        engine_group = QGroupBox("Engine")
        engine_layout = QFormLayout(engine_group)
        engine_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        engine_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self.thrust_sl = QDoubleSpinBox()
        self.thrust_sl.setRange(0, 100000)
        self.thrust_sl.setValue(500)
        self.thrust_sl.setSuffix(" kN")
        self.thrust_sl.setDecimals(0)
        self.thrust_sl.valueChanged.connect(self._emit_change)
        engine_layout.addRow("Thrust (SL):", self.thrust_sl)
        
        self.thrust_vac = QDoubleSpinBox()
        self.thrust_vac.setRange(0, 100000)
        self.thrust_vac.setValue(550)
        self.thrust_vac.setSuffix(" kN")
        self.thrust_vac.setDecimals(0)
        self.thrust_vac.valueChanged.connect(self._emit_change)
        engine_layout.addRow("Thrust (Vac):", self.thrust_vac)
        
        self.isp_sl = QDoubleSpinBox()
        self.isp_sl.setRange(0, 500)
        self.isp_sl.setValue(280)
        self.isp_sl.setSuffix(" s")
        self.isp_sl.setDecimals(0)
        self.isp_sl.valueChanged.connect(self._emit_change)
        engine_layout.addRow("Isp (SL):", self.isp_sl)
        
        self.isp_vac = QDoubleSpinBox()
        self.isp_vac.setRange(100, 500)
        self.isp_vac.setValue(310)
        self.isp_vac.setSuffix(" s")
        self.isp_vac.setDecimals(0)
        self.isp_vac.valueChanged.connect(self._emit_change)
        engine_layout.addRow("Isp (Vac):", self.isp_vac)
        
        self.num_engines = QSpinBox()
        self.num_engines.setRange(1, 50)
        self.num_engines.setValue(1)
        self.num_engines.valueChanged.connect(self._emit_change)
        engine_layout.addRow("Engines:", self.num_engines)
        
        layout.addWidget(engine_group)
        
        # Calculated values
        calc_group = QGroupBox("Calculated")
        calc_layout = QFormLayout(calc_group)
        calc_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.mass_ratio_label = QLabel("--")
        self.mass_ratio_label.setObjectName("calcValue")
        calc_layout.addRow("Mass Ratio:", self.mass_ratio_label)
        
        self.delta_v_label = QLabel("--")
        self.delta_v_label.setObjectName("calcValue")
        calc_layout.addRow("Stage ΔV:", self.delta_v_label)
        
        layout.addWidget(calc_group)
        layout.addStretch()
    
    def _emit_change(self):
        self._update_calculations()
        self.stage_changed.emit()
    
    def _update_calculations(self):
        """Update calculated values display."""
        dry = self.dry_mass.value()
        prop = self.prop_mass.value()
        isp_vac = self.isp_vac.value()
        
        if dry > 0:
            mass_ratio = (dry + prop) / dry
            self.mass_ratio_label.setText(f"{mass_ratio:.2f}")
            
            import math
            delta_v = isp_vac * 9.80665 * math.log(mass_ratio)
            self.delta_v_label.setText(f"{delta_v:.0f} m/s")
    
    def get_stage_config(self) -> dict:
        """Get stage configuration as dictionary."""
        return {
            'name': f'Stage {self.stage_num}',
            'dry_mass': self.dry_mass.value(),
            'propellant_mass': self.prop_mass.value(),
            'thrust_sl': self.thrust_sl.value() * 1000,  # kN to N
            'thrust_vac': self.thrust_vac.value() * 1000,
            'isp_sl': self.isp_sl.value(),
            'isp_vac': self.isp_vac.value(),
            'num_engines': self.num_engines.value()
        }
    
    def set_stage_config(self, config: dict):
        """Set stage configuration from dictionary."""
        self.dry_mass.setValue(config.get('dry_mass', 5000))
        self.prop_mass.setValue(config.get('propellant_mass', 50000))
        self.thrust_sl.setValue(config.get('thrust_sl', 500000) / 1000)
        self.thrust_vac.setValue(config.get('thrust_vac', 550000) / 1000)
        self.isp_sl.setValue(config.get('isp_sl', 280))
        self.isp_vac.setValue(config.get('isp_vac', 310))
        self.num_engines.setValue(config.get('num_engines', 1))
        self._update_calculations()


class MultiStageWidget(QWidget):
    """
    Complete multi-stage vehicle configuration widget.
    
    Features:
    - Add/remove stages
    - Load preset vehicles (Falcon 9, Saturn V)
    - Configure each stage's mass and engine properties
    - Real-time delta-v breakdown
    - Payload mass input
    """
    
    vehicle_changed = pyqtSignal(object)  # Emits MultiStageVehicle
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_cards: list[StageConfigCard] = []
        self._setup_ui()
        self._add_default_stages()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # === Top Controls ===
        controls = QHBoxLayout()
        
        # Preset selector
        controls.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Falcon 9 (2-stage)",
            "Saturn V (3-stage)",
            "Small Launcher (2-stage)",
            "Heavy Lift (3-stage)"
        ])
        self.preset_combo.currentIndexChanged.connect(self._load_preset)
        controls.addWidget(self.preset_combo)
        
        controls.addStretch()
        
        # Add stage button
        self.add_stage_btn = QPushButton("+ Add Stage")
        self.add_stage_btn.setObjectName("addStageBtn")
        self.add_stage_btn.clicked.connect(self._add_stage)
        controls.addWidget(self.add_stage_btn)
        
        layout.addLayout(controls)
        
        # === Payload Mass ===
        payload_layout = QHBoxLayout()
        payload_layout.addWidget(QLabel("Payload Mass:"))
        self.payload_spin = QDoubleSpinBox()
        self.payload_spin.setRange(0, 500000)
        self.payload_spin.setValue(5000)
        self.payload_spin.setSuffix(" kg")
        self.payload_spin.setDecimals(0)
        self.payload_spin.valueChanged.connect(self._update_vehicle)
        payload_layout.addWidget(self.payload_spin)
        payload_layout.addStretch()
        layout.addLayout(payload_layout)
        
        # === Stage Cards Container ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.stages_container = QWidget()
        self.stages_layout = QHBoxLayout(self.stages_container)
        self.stages_layout.setContentsMargins(0, 0, 0, 0)
        self.stages_layout.setSpacing(10)
        self.stages_layout.addStretch()
        
        scroll.setWidget(self.stages_container)
        layout.addWidget(scroll, 1)
        
        # === Summary Panel ===
        summary_group = QGroupBox("Vehicle Summary")
        summary_layout = QGridLayout(summary_group)
        summary_layout.setContentsMargins(16, 20, 16, 16)
        summary_layout.setSpacing(12)
        
        summary_layout.addWidget(QLabel("Total Stages:"), 0, 0)
        self.total_stages_label = QLabel("0")
        self.total_stages_label.setObjectName("summaryValue")
        summary_layout.addWidget(self.total_stages_label, 0, 1)
        
        summary_layout.addWidget(QLabel("Total Mass:"), 0, 2)
        self.total_mass_label = QLabel("-- kg")
        self.total_mass_label.setObjectName("summaryValue")
        summary_layout.addWidget(self.total_mass_label, 0, 3)
        
        summary_layout.addWidget(QLabel("Total ΔV:"), 1, 0)
        self.total_dv_label = QLabel("-- m/s")
        self.total_dv_label.setObjectName("summaryValue")
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        self.total_dv_label.setFont(font)
        summary_layout.addWidget(self.total_dv_label, 1, 1)
        
        summary_layout.addWidget(QLabel("To LEO:"), 1, 2)
        self.leo_capability_label = QLabel("--")
        self.leo_capability_label.setObjectName("summaryValue")
        summary_layout.addWidget(self.leo_capability_label, 1, 3)
        
        layout.addWidget(summary_group)
        
        # === Delta-V Breakdown Table ===
        breakdown_group = QGroupBox("Stage ΔV Breakdown")
        breakdown_layout = QVBoxLayout(breakdown_group)
        
        self.dv_table = QTableWidget()
        self.dv_table.setColumnCount(4)
        self.dv_table.setHorizontalHeaderLabels(["Stage", "ΔV (m/s)", "Cumulative", "% of Total"])
        self.dv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dv_table.setMinimumHeight(200) # Give it more room
        self.dv_table.verticalHeader().setVisible(False)
        self.dv_table.verticalHeader().setDefaultSectionSize(32)
        breakdown_layout.addWidget(self.dv_table)
        
        # Progress bar for total delta-v
        self.dv_progress = QProgressBar()
        self.dv_progress.setRange(0, 10000)  # m/s to LEO
        self.dv_progress.setFixedHeight(24)
        self.dv_progress.setFormat(" ΔV to LEO: %v / 9400 m/s")
        breakdown_layout.addWidget(self.dv_progress)
        
        layout.addWidget(breakdown_group)
    
    def _add_default_stages(self):
        """Add default 2-stage configuration."""
        # Stage 1
        self._add_stage()
        self.stage_cards[0].dry_mass.setValue(25000)
        self.stage_cards[0].prop_mass.setValue(400000)
        self.stage_cards[0].thrust_sl.setValue(7000)
        self.stage_cards[0].thrust_vac.setValue(7600)
        self.stage_cards[0].isp_sl.setValue(282)
        self.stage_cards[0].isp_vac.setValue(311)
        self.stage_cards[0].num_engines.setValue(9)
        
        # Stage 2
        self._add_stage()
        self.stage_cards[1].dry_mass.setValue(4000)
        self.stage_cards[1].prop_mass.setValue(100000)
        self.stage_cards[1].thrust_sl.setValue(0)
        self.stage_cards[1].thrust_vac.setValue(930)
        self.stage_cards[1].isp_sl.setValue(0)
        self.stage_cards[1].isp_vac.setValue(348)
        self.stage_cards[1].num_engines.setValue(1)
        
        self._update_vehicle()
    
    def _add_stage(self):
        """Add a new stage."""
        stage_num = len(self.stage_cards) + 1
        card = StageConfigCard(stage_num)
        card.stage_changed.connect(self._update_vehicle)
        card.remove_requested.connect(self._remove_stage)
        
        # Insert before the stretch
        self.stages_layout.insertWidget(self.stages_layout.count() - 1, card)
        self.stage_cards.append(card)
        self._update_vehicle()
    
    def _remove_stage(self, stage_num: int):
        """Remove a stage."""
        if len(self.stage_cards) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "Vehicle must have at least one stage.")
            return
        
        # Find and remove the card
        for i, card in enumerate(self.stage_cards):
            if card.stage_num == stage_num:
                self.stages_layout.removeWidget(card)
                card.deleteLater()
                self.stage_cards.pop(i)
                break
        
        # Renumber remaining stages
        for i, card in enumerate(self.stage_cards):
            card.stage_num = i + 1
            card.findChild(QLabel, "").setText(f"Stage {i + 1}")
        
        self._update_vehicle()
    
    def _load_preset(self, index: int):
        """Load a preset vehicle configuration."""
        if index == 0:  # Custom
            return
        
        # Clear existing stages
        for card in self.stage_cards[:]:
            self.stages_layout.removeWidget(card)
            card.deleteLater()
        self.stage_cards.clear()
        
        if index == 1:  # Falcon 9
            vehicle = create_falcon_9_like()
        elif index == 2:  # Saturn V
            vehicle = create_saturn_v_like()
        elif index == 3:  # Small Launcher
            configs = [
                {'dry_mass': 2000, 'propellant_mass': 20000, 'thrust_sl': 300000, 'thrust_vac': 330000, 'isp_sl': 270, 'isp_vac': 295, 'num_engines': 1},
                {'dry_mass': 500, 'propellant_mass': 5000, 'thrust_sl': 0, 'thrust_vac': 50000, 'isp_sl': 0, 'isp_vac': 320, 'num_engines': 1}
            ]
            for i, cfg in enumerate(configs):
                self._add_stage()
                self.stage_cards[i].set_stage_config(cfg)
            self._update_vehicle()
            return
        elif index == 4:  # Heavy Lift
            configs = [
                {'dry_mass': 100000, 'propellant_mass': 2000000, 'thrust_sl': 30000000, 'thrust_vac': 35000000, 'isp_sl': 265, 'isp_vac': 300, 'num_engines': 5},
                {'dry_mass': 30000, 'propellant_mass': 400000, 'thrust_sl': 0, 'thrust_vac': 5000000, 'isp_sl': 0, 'isp_vac': 420, 'num_engines': 5},
                {'dry_mass': 10000, 'propellant_mass': 100000, 'thrust_sl': 0, 'thrust_vac': 1000000, 'isp_sl': 0, 'isp_vac': 420, 'num_engines': 1}
            ]
            for i, cfg in enumerate(configs):
                self._add_stage()
                self.stage_cards[i].set_stage_config(cfg)
            self._update_vehicle()
            return
        else:
            return
        
        # Load from MultiStageVehicle object
        for i, stage in enumerate(vehicle.stages):
            self._add_stage()
            config = {
                'dry_mass': stage.dry_mass,
                'propellant_mass': stage.propellant_mass,
                'thrust_sl': stage.engine.thrust_sl,
                'thrust_vac': stage.engine.thrust_vac,
                'isp_sl': stage.engine.isp_sl,
                'isp_vac': stage.engine.isp_vac,
                'num_engines': stage.engine.num_engines
            }
            self.stage_cards[i].set_stage_config(config)
        
        self._update_vehicle()
    
    def _update_vehicle(self):
        """Update vehicle model and summary display."""
        if not self.stage_cards:
            return
        
        # Build vehicle from current configuration
        configs = [card.get_stage_config() for card in self.stage_cards]
        payload = self.payload_spin.value()
        
        try:
            vehicle = create_custom_vehicle(configs, payload, "Custom Vehicle")
            
            # Update summary
            self.total_stages_label.setText(str(len(vehicle.stages)))
            self.total_mass_label.setText(f"{vehicle.total_mass:,.0f} kg")
            
            total_dv = vehicle.get_total_delta_v()
            self.total_dv_label.setText(f"{total_dv:,.0f} m/s")
            
            # LEO capability estimate
            leo_dv = 9400  # m/s typical
            if total_dv >= leo_dv:
                self.leo_capability_label.setText("✓ LEO Capable")
                self.leo_capability_label.setStyleSheet("color: #00ff88;")
            else:
                deficit = leo_dv - total_dv
                self.leo_capability_label.setText(f"Need {deficit:.0f} m/s more")
                self.leo_capability_label.setStyleSheet("color: #ff6b6b;")
            
            # Update progress bar
            self.dv_progress.setValue(min(int(total_dv), 10000))
            
            # Update breakdown table
            breakdown = vehicle.get_stage_delta_v_breakdown()
            self.dv_table.setRowCount(len(breakdown))
            
            cumulative = 0
            for i, (name, dv) in enumerate(breakdown):
                cumulative += dv
                pct = (dv / total_dv * 100) if total_dv > 0 else 0
                
                self.dv_table.setItem(i, 0, QTableWidgetItem(name))
                self.dv_table.setItem(i, 1, QTableWidgetItem(f"{dv:,.0f}"))
                self.dv_table.setItem(i, 2, QTableWidgetItem(f"{cumulative:,.0f}"))
                self.dv_table.setItem(i, 3, QTableWidgetItem(f"{pct:.1f}%"))
            
            # Emit signal
            self.vehicle_changed.emit(vehicle)
            
        except Exception as e:
            print(f"Error building vehicle: {e}")
    
    def get_vehicle(self) -> MultiStageVehicle | None:
        """Get the current vehicle configuration."""
        if not self.stage_cards:
            return None
        
        configs = [card.get_stage_config() for card in self.stage_cards]
        payload = self.payload_spin.value()
        
        try:
            return create_custom_vehicle(configs, payload, "Custom Vehicle")
        except Exception:
            return None

