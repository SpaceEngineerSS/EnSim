"""
Propellant Database Editor.

Provides:
- Form to add custom species with NASA coefficients
- Persistence to user_propellants.json
- Integration with chemistry engine

Phase 7: Professional UX feature.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import json

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QDoubleSpinBox, QTextEdit, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QWidget, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CustomPropellant:
    """User-defined propellant species."""
    name: str
    formula: str
    molecular_weight: float  # g/mol
    enthalpy_formation: float  # kJ/mol
    phase: str = "G"  # G=gas, L=liquid, S=solid
    
    # NASA 7-term polynomial coefficients (high temp range)
    nasa_high: List[float] = None  # 7 coefficients
    temp_high_range: List[float] = None  # [Tmin, Tmax]
    
    # NASA 7-term polynomial coefficients (low temp range)
    nasa_low: List[float] = None  # 7 coefficients
    temp_low_range: List[float] = None  # [Tmin, Tmax]
    
    def __post_init__(self):
        if self.nasa_high is None:
            self.nasa_high = [0.0] * 7
        if self.nasa_low is None:
            self.nasa_low = [0.0] * 7
        if self.temp_high_range is None:
            self.temp_high_range = [1000.0, 5000.0]
        if self.temp_low_range is None:
            self.temp_low_range = [300.0, 1000.0]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CustomPropellant':
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# Propellant Database Manager
# =============================================================================

class PropellantDatabase:
    """
    Manages custom propellant database.
    
    Stores in user's home directory for persistence.
    """
    
    def __init__(self):
        self._propellants: Dict[str, CustomPropellant] = {}
        self._db_path = self._get_db_path()
        self._load()
    
    def _get_db_path(self) -> Path:
        """Get path to user's propellant database."""
        # Use .ensim directory in user home
        ensim_dir = Path.home() / ".ensim"
        ensim_dir.mkdir(exist_ok=True)
        return ensim_dir / "user_propellants.json"
    
    def _load(self):
        """Load propellants from disk."""
        if self._db_path.exists():
            try:
                with open(self._db_path, 'r') as f:
                    data = json.load(f)
                
                for name, prop_data in data.items():
                    self._propellants[name] = CustomPropellant.from_dict(prop_data)
                    
            except Exception as e:
                print(f"Error loading propellant database: {e}")
    
    def save(self):
        """Save propellants to disk."""
        try:
            data = {name: prop.to_dict() for name, prop in self._propellants.items()}
            
            with open(self._db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving propellant database: {e}")
    
    def add(self, propellant: CustomPropellant):
        """Add or update a propellant."""
        self._propellants[propellant.name] = propellant
        self.save()
    
    def remove(self, name: str):
        """Remove a propellant."""
        if name in self._propellants:
            del self._propellants[name]
            self.save()
    
    def get(self, name: str) -> Optional[CustomPropellant]:
        """Get propellant by name."""
        return self._propellants.get(name)
    
    def get_all(self) -> List[CustomPropellant]:
        """Get all propellants."""
        return list(self._propellants.values())
    
    def get_names(self) -> List[str]:
        """Get all propellant names."""
        return list(self._propellants.keys())


# Global database instance
_propellant_db: Optional[PropellantDatabase] = None


def get_propellant_database() -> PropellantDatabase:
    """Get the global propellant database instance."""
    global _propellant_db
    if _propellant_db is None:
        _propellant_db = PropellantDatabase()
    return _propellant_db


# =============================================================================
# Propellant Editor Widget
# =============================================================================

class PropellantEditorWidget(QWidget):
    """
    Widget for editing custom propellants.
    
    Provides form to add/edit species with NASA coefficients.
    """
    
    propellant_added = pyqtSignal(str)  # Emits name when propellant added
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = get_propellant_database()
        self._setup_ui()
        self._load_list()
    
    def _setup_ui(self):
        """Build the editor UI."""
        layout = QHBoxLayout(self)
        
        # Left: List of existing propellants
        list_widget = QWidget()
        list_widget.setMaximumWidth(250)
        list_layout = QVBoxLayout(list_widget)
        
        list_layout.addWidget(QLabel("Custom Propellants:"))
        
        self.prop_table = QTableWidget()
        self.prop_table.setColumnCount(2)
        self.prop_table.setHorizontalHeaderLabels(["Name", "Formula"])
        self.prop_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prop_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.prop_table.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.prop_table)
        
        # List buttons
        list_btn_layout = QHBoxLayout()
        
        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self._new_propellant)
        list_btn_layout.addWidget(self.new_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_propellant)
        list_btn_layout.addWidget(self.delete_btn)
        
        list_layout.addLayout(list_btn_layout)
        layout.addWidget(list_widget)
        
        # Right: Editor form
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        # Basic properties
        basic_group = QGroupBox("Basic Properties")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., RP-1")
        basic_layout.addRow("Name:", self.name_edit)
        
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g., C12H24")
        basic_layout.addRow("Formula:", self.formula_edit)
        
        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setRange(1, 1000)
        self.mw_spin.setDecimals(3)
        self.mw_spin.setSuffix(" g/mol")
        basic_layout.addRow("Molecular Weight:", self.mw_spin)
        
        self.hf_spin = QDoubleSpinBox()
        self.hf_spin.setRange(-1000, 1000)
        self.hf_spin.setDecimals(2)
        self.hf_spin.setSuffix(" kJ/mol")
        basic_layout.addRow("Enthalpy of Formation:", self.hf_spin)
        
        form_layout.addWidget(basic_group)
        
        # NASA coefficients
        nasa_group = QGroupBox("NASA 7-Term Coefficients")
        nasa_layout = QVBoxLayout(nasa_group)
        
        nasa_layout.addWidget(QLabel(
            "Paste coefficients from NASA CEA database.\n"
            "Format: 7 values per line (high temp, then low temp)"
        ))
        
        self.nasa_text = QTextEdit()
        self.nasa_text.setPlaceholderText(
            "a1 a2 a3 a4 a5 a6 a7  (1000-5000 K)\n"
            "a1 a2 a3 a4 a5 a6 a7  (300-1000 K)"
        )
        self.nasa_text.setMaximumHeight(100)
        nasa_layout.addWidget(self.nasa_text)
        
        form_layout.addWidget(nasa_group)
        
        # Save button
        self.save_btn = QPushButton("💾 Save Propellant")
        self.save_btn.setObjectName("runButton")
        self.save_btn.clicked.connect(self._save_propellant)
        form_layout.addWidget(self.save_btn)
        
        form_layout.addStretch()
        layout.addWidget(form_widget, stretch=1)
    
    def _load_list(self):
        """Load propellant list from database."""
        self.prop_table.setRowCount(0)
        
        for prop in self._db.get_all():
            row = self.prop_table.rowCount()
            self.prop_table.insertRow(row)
            self.prop_table.setItem(row, 0, QTableWidgetItem(prop.name))
            self.prop_table.setItem(row, 1, QTableWidgetItem(prop.formula))
    
    def _on_selection_changed(self):
        """Handle selection change in list."""
        rows = self.prop_table.selectedItems()
        if rows:
            name = self.prop_table.item(rows[0].row(), 0).text()
            self._load_propellant(name)
    
    def _load_propellant(self, name: str):
        """Load propellant data into form."""
        prop = self._db.get(name)
        if not prop:
            return
        
        self.name_edit.setText(prop.name)
        self.formula_edit.setText(prop.formula)
        self.mw_spin.setValue(prop.molecular_weight)
        self.hf_spin.setValue(prop.enthalpy_formation)
        
        # Format NASA coefficients
        high_str = " ".join(f"{c:.6e}" for c in prop.nasa_high)
        low_str = " ".join(f"{c:.6e}" for c in prop.nasa_low)
        self.nasa_text.setText(f"{high_str}\n{low_str}")
    
    def _new_propellant(self):
        """Clear form for new propellant."""
        self.name_edit.clear()
        self.formula_edit.clear()
        self.mw_spin.setValue(0)
        self.hf_spin.setValue(0)
        self.nasa_text.clear()
        self.prop_table.clearSelection()
    
    def _delete_propellant(self):
        """Delete selected propellant."""
        rows = self.prop_table.selectedItems()
        if not rows:
            return
        
        name = self.prop_table.item(rows[0].row(), 0).text()
        
        reply = QMessageBox.question(
            self, "Delete Propellant",
            f"Delete propellant '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._db.remove(name)
            self._load_list()
            self._new_propellant()
    
    def _save_propellant(self):
        """Save current propellant."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Name is required.")
            return
        
        formula = self.formula_edit.text().strip()
        if not formula:
            QMessageBox.warning(self, "Error", "Formula is required.")
            return
        
        # Parse NASA coefficients
        nasa_high = [0.0] * 7
        nasa_low = [0.0] * 7
        
        nasa_text = self.nasa_text.toPlainText().strip()
        if nasa_text:
            lines = nasa_text.split('\n')
            try:
                if len(lines) >= 1:
                    nasa_high = [float(x) for x in lines[0].split()[:7]]
                if len(lines) >= 2:
                    nasa_low = [float(x) for x in lines[1].split()[:7]]
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid NASA coefficient format.")
                return
        
        # Create propellant
        prop = CustomPropellant(
            name=name,
            formula=formula,
            molecular_weight=self.mw_spin.value(),
            enthalpy_formation=self.hf_spin.value(),
            nasa_high=nasa_high,
            nasa_low=nasa_low
        )
        
        self._db.add(prop)
        self._load_list()
        self.propellant_added.emit(name)
        
        QMessageBox.information(self, "Saved", f"Propellant '{name}' saved.")


class PropellantEditorDialog(QDialog):
    """Dialog wrapper for propellant editor."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Propellant Database Editor")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(self)
        
        header = QLabel("🧪 Custom Propellant Database")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00a8ff;")
        layout.addWidget(header)
        
        self.editor = PropellantEditorWidget()
        layout.addWidget(self.editor)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Propellant Editor...")
    print("=" * 50)
    
    db = get_propellant_database()
    
    # Add test propellant
    prop = CustomPropellant(
        name="RP-1",
        formula="C12H24",
        molecular_weight=170.0,
        enthalpy_formation=-250.0,
        nasa_high=[1.5e1, 1.2e-2, -3.5e-6, 4.2e-10, -1.8e-14, -2.5e4, 1.0e1],
        nasa_low=[2.0e0, 5.0e-2, -2.0e-5, 6.0e-9, -8.0e-13, -2.3e4, 2.5e1]
    )
    db.add(prop)
    
    print(f"Database path: {db._db_path}")
    print(f"Propellants: {db.get_names()}")
    print("✓ Propellant editor module ready!")
