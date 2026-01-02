"""
Propellant Presets Selection Widget.

Provides a comprehensive propellant selector with:
- 17+ preset combinations
- Category filtering
- Performance preview
- Auto-fill of O/F ratio and properties
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QComboBox, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit
)
from PyQt6.QtGui import QFont

from ...data.propellant_presets import (
    PROPELLANT_PRESETS, PropellantCategory, ToxicityLevel,
    get_preset, get_presets_by_category, get_all_preset_names,
    get_non_toxic_presets
)


class PropellantPresetWidget(QWidget):
    """
    Propellant preset selection and information widget.
    
    Signals:
        preset_selected: Emitted when a preset is selected, with preset data
    """
    
    preset_selected = pyqtSignal(object)  # Emits PropellantPreset
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._populate_presets()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt6.QtWidgets import QScrollArea, QFrame
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # === Category Filter ===
        filter_layout = QHBoxLayout()
        lbl_cat = QLabel("Category:")
        lbl_cat.setMinimumWidth(80)
        filter_layout.addWidget(lbl_cat)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems([
            "All Propellants",
            "Cryogenic",
            "Hydrocarbon",
            "Storable",
            "Green/Low-Toxicity",
            "Monopropellant",
            "Exotic"
        ])
        self.category_combo.currentIndexChanged.connect(self._filter_presets)
        filter_layout.addWidget(self.category_combo, 1)
        
        # Quick filter buttons
        self.high_isp_btn = QPushButton("High Isp")
        self.high_isp_btn.setCheckable(True)
        self.high_isp_btn.clicked.connect(self._filter_presets)
        filter_layout.addWidget(self.high_isp_btn)
        
        self.non_toxic_btn = QPushButton("Non-Toxic")
        self.non_toxic_btn.setCheckable(True)
        self.non_toxic_btn.clicked.connect(self._filter_presets)
        filter_layout.addWidget(self.non_toxic_btn)
        
        layout.addLayout(filter_layout)
        
        # === Preset Selector ===
        selector_layout = QHBoxLayout()
        lbl_prop = QLabel("Propellant:")
        lbl_prop.setMinimumWidth(80)
        selector_layout.addWidget(lbl_prop)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(200)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        selector_layout.addWidget(self.preset_combo, 1)
        
        self.apply_btn = QPushButton("Apply Configuration")
        self.apply_btn.setObjectName("applyPresetBtn")
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.clicked.connect(self._apply_preset)
        selector_layout.addWidget(self.apply_btn)
        
        layout.addLayout(selector_layout)
        
        # === Performance Preview ===
        perf_group = QGroupBox("Performance Data")
        perf_layout = QGridLayout(perf_group)
        perf_layout.setSpacing(10)
        perf_layout.setColumnStretch(1, 1)
        perf_layout.setColumnStretch(3, 1)
        
        # Create labels for performance metrics
        metrics = [
            ("Vacuum Isp:", "isp_vac", "s"),
            ("Sea-Level Isp:", "isp_sl", "s"),
            ("C*:", "cstar", "m/s"),
            ("Chamber Temp:", "t_chamber", "K"),
            ("Optimal O/F:", "of_ratio", ""),
            ("O/F Range:", "of_range", ""),
            ("Bulk Density:", "density", "kg/m³"),
            ("γ (Gamma):", "gamma", ""),
        ]
        
        self.metric_labels = {}
        for i, (text, key, unit) in enumerate(metrics):
            row, col = i // 2, (i % 2) * 2
            
            label_text = QLabel(text)
            label_text.setMinimumWidth(110)
            label_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            perf_layout.addWidget(label_text, row, col)
            
            label = QLabel("--")
            label.setObjectName("calcValue")
            self.metric_labels[key] = label
            perf_layout.addWidget(label, row, col + 1)
            
            if unit:
                unit_label = QLabel(unit)
                unit_label.setStyleSheet("color: #666666; font-size: 10px;")
                perf_layout.addWidget(unit_label, row, col + 1, Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(perf_group)
        
        # === Component Details ===
        comp_group = QGroupBox("Components")
        comp_layout = QGridLayout(comp_group)
        comp_layout.setSpacing(10)
        
        lbl_fuel = QLabel("Fuel:")
        lbl_fuel.setMinimumWidth(110)
        comp_layout.addWidget(lbl_fuel, 0, 0, Qt.AlignmentFlag.AlignRight)
        self.fuel_name = QLabel("--")
        self.fuel_name.setObjectName("summaryValue")
        comp_layout.addWidget(self.fuel_name, 0, 1)
        
        comp_layout.addWidget(QLabel("Formula:"), 0, 2, Qt.AlignmentFlag.AlignRight)
        self.fuel_formula = QLabel("--")
        self.fuel_formula.setObjectName("calcValue")
        comp_layout.addWidget(self.fuel_formula, 0, 3)
        
        lbl_ox = QLabel("Oxidizer:")
        comp_layout.addWidget(lbl_ox, 1, 0, Qt.AlignmentFlag.AlignRight)
        self.ox_name = QLabel("--")
        self.ox_name.setObjectName("summaryValue")
        comp_layout.addWidget(self.ox_name, 1, 1)
        
        comp_layout.addWidget(QLabel("Formula:"), 1, 2, Qt.AlignmentFlag.AlignRight)
        self.ox_formula = QLabel("--")
        self.ox_formula.setObjectName("calcValue")
        comp_layout.addWidget(self.ox_formula, 1, 3)
        
        lbl_tox = QLabel("Toxicity:")
        comp_layout.addWidget(lbl_tox, 2, 0, Qt.AlignmentFlag.AlignRight)
        self.toxicity_label = QLabel("--")
        comp_layout.addWidget(self.toxicity_label, 2, 1, 1, 3)
        
        layout.addWidget(comp_group)
        
        # === Applications ===
        apps_group = QGroupBox("Applications & Notes")
        apps_layout = QVBoxLayout(apps_group)
        apps_layout.setSpacing(8)
        
        self.applications_label = QLabel("--")
        self.applications_label.setWordWrap(True)
        self.applications_label.setObjectName("summaryValue")
        apps_layout.addWidget(self.applications_label)
        
        self.notes_label = QLabel("--")
        self.notes_label.setWordWrap(True)
        self.notes_label.setObjectName("notesText")
        apps_layout.addWidget(self.notes_label)
        
        layout.addWidget(apps_group)
        
        # === Comparison Table ===
        compare_group = QGroupBox("Quick Comparison")
        compare_layout = QVBoxLayout(compare_group)
        
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(5)
        self.compare_table.setHorizontalHeaderLabels([
            "Propellant", "Isp (vac)", "Density", "O/F", "Category"
        ])
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.compare_table.setMinimumHeight(240) # Increased height
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.verticalHeader().setDefaultSectionSize(30)
        self.compare_table.cellClicked.connect(self._on_table_click)
        compare_layout.addWidget(self.compare_table)
        
        layout.addWidget(compare_group)
        layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
    
    def _populate_presets(self):
        """Populate the preset combo box."""
        self.preset_combo.clear()
        self.preset_combo.addItem("-- Select Propellant --", None)
        
        for name in get_all_preset_names():
            preset = get_preset(name)
            if preset:
                display_name = f"{preset.name} ({preset.isp_vacuum:.0f}s)"
                self.preset_combo.addItem(display_name, name)
        
        self._update_comparison_table()
    
    def _filter_presets(self):
        """Filter presets based on category and quick filters."""
        self.preset_combo.clear()
        self.preset_combo.addItem("-- Select Propellant --", None)
        
        category_idx = self.category_combo.currentIndex()
        high_isp = self.high_isp_btn.isChecked()
        non_toxic = self.non_toxic_btn.isChecked()
        
        # Get category filter
        category_map = {
            0: None,  # All
            1: PropellantCategory.CRYOGENIC,
            2: PropellantCategory.HYDROCARBON,
            3: PropellantCategory.STORABLE,
            4: PropellantCategory.GREEN,
            5: PropellantCategory.MONOPROPELLANT,
            6: PropellantCategory.EXOTIC
        }
        category = category_map.get(category_idx)
        
        # Filter presets
        if non_toxic:
            presets = get_non_toxic_presets()
        elif category:
            presets = get_presets_by_category(category)
        else:
            presets = list(PROPELLANT_PRESETS.values())
        
        # Apply high Isp filter
        if high_isp:
            presets = [p for p in presets if p.isp_vacuum >= 350]
        
        # Populate combo
        for preset in sorted(presets, key=lambda p: -p.isp_vacuum):
            display_name = f"{preset.name} ({preset.isp_vacuum:.0f}s)"
            self.preset_combo.addItem(display_name, preset.name.replace("/", "_").replace(" ", "_").replace("-", ""))
        
        self._update_comparison_table(presets)
    
    def _update_comparison_table(self, presets=None):
        """Update the comparison table."""
        if presets is None:
            presets = list(PROPELLANT_PRESETS.values())
        
        self.compare_table.setRowCount(min(len(presets), 10))
        
        for i, preset in enumerate(sorted(presets, key=lambda p: -p.isp_vacuum)[:10]):
            self.compare_table.setItem(i, 0, QTableWidgetItem(preset.name))
            self.compare_table.setItem(i, 1, QTableWidgetItem(f"{preset.isp_vacuum:.0f} s"))
            self.compare_table.setItem(i, 2, QTableWidgetItem(f"{preset.density_bulk:.0f}"))
            self.compare_table.setItem(i, 3, QTableWidgetItem(f"{preset.of_ratio_optimal:.1f}"))
            self.compare_table.setItem(i, 4, QTableWidgetItem(preset.category.name))
    
    def _on_table_click(self, row: int, col: int):
        """Handle table row click to select preset."""
        name_item = self.compare_table.item(row, 0)
        if name_item:
            # Find and select in combo
            for i in range(self.preset_combo.count()):
                if name_item.text() in self.preset_combo.itemText(i):
                    self.preset_combo.setCurrentIndex(i)
                    break
    
    def _on_preset_changed(self, index: int):
        """Handle preset selection change."""
        preset_key = self.preset_combo.itemData(index)
        if not preset_key:
            self._clear_display()
            return
        
        preset = get_preset(preset_key)
        if not preset:
            # Try finding by iterating
            for key, p in PROPELLANT_PRESETS.items():
                if p.name in self.preset_combo.itemText(index):
                    preset = p
                    break
        
        if preset:
            self._update_display(preset)
    
    def _update_display(self, preset):
        """Update all display fields with preset data."""
        # Performance metrics
        self.metric_labels['isp_vac'].setText(f"{preset.isp_vacuum:.0f} s")
        self.metric_labels['isp_sl'].setText(f"{preset.isp_sea_level:.0f} s")
        self.metric_labels['cstar'].setText(f"{preset.c_star:.0f} m/s")
        self.metric_labels['t_chamber'].setText(f"{preset.chamber_temp:.0f} K")
        self.metric_labels['of_ratio'].setText(f"{preset.of_ratio_optimal:.2f}")
        self.metric_labels['of_range'].setText(
            f"{preset.of_ratio_range[0]:.1f} - {preset.of_ratio_range[1]:.1f}"
        )
        self.metric_labels['density'].setText(f"{preset.density_bulk:.0f} kg/m³")
        self.metric_labels['gamma'].setText(f"{preset.gamma:.3f}")
        
        # Components
        self.fuel_name.setText(preset.fuel.name)
        self.fuel_formula.setText(preset.fuel.formula)
        self.ox_name.setText(preset.oxidizer.name)
        self.ox_formula.setText(preset.oxidizer.formula)
        
        # Toxicity
        fuel_tox = preset.fuel.toxicity.name
        ox_tox = preset.oxidizer.toxicity.name
        max_tox = max(preset.fuel.toxicity.value, preset.oxidizer.toxicity.value)
        
        toxicity_text = f"Fuel: {fuel_tox}, Oxidizer: {ox_tox}"
        self.toxicity_label.setText(toxicity_text)
        
        # Color code toxicity
        if max_tox <= ToxicityLevel.LOW.value:
            self.toxicity_label.setStyleSheet("color: #00ff88;")
        elif max_tox <= ToxicityLevel.MODERATE.value:
            self.toxicity_label.setStyleSheet("color: #ffcc00;")
        else:
            self.toxicity_label.setStyleSheet("color: #ff6b6b;")
        
        # Applications
        self.applications_label.setText("Applications: " + ", ".join(preset.applications))
        self.notes_label.setText("Notes: " + preset.notes)
    
    def _clear_display(self):
        """Clear all display fields."""
        for label in self.metric_labels.values():
            label.setText("--")
        self.fuel_name.setText("--")
        self.fuel_formula.setText("--")
        self.ox_name.setText("--")
        self.ox_formula.setText("--")
        self.toxicity_label.setText("--")
        self.applications_label.setText("--")
        self.notes_label.setText("--")
    
    def _apply_preset(self):
        """Apply the selected preset."""
        index = self.preset_combo.currentIndex()
        preset_key = self.preset_combo.itemData(index)
        
        if not preset_key:
            return
        
        preset = get_preset(preset_key)
        if not preset:
            # Try finding by name
            for key, p in PROPELLANT_PRESETS.items():
                if p.name in self.preset_combo.itemText(index):
                    preset = p
                    break
        
        if preset:
            self.preset_selected.emit(preset)
    
    def get_selected_preset(self):
        """Get the currently selected preset."""
        index = self.preset_combo.currentIndex()
        preset_key = self.preset_combo.itemData(index)
        if preset_key:
            return get_preset(preset_key)
        return None
    
    def get_of_ratio(self) -> float:
        """Get optimal O/F ratio of selected preset."""
        preset = self.get_selected_preset()
        return preset.of_ratio_optimal if preset else 2.7
    
    def get_gamma(self) -> float:
        """Get gamma of selected preset."""
        preset = self.get_selected_preset()
        return preset.gamma if preset else 1.2

