# Main login styles
LOGIN_STYLES = """
    QWidget {
        background-color: #202124;
        font-family: "Consolas", "Courier New", monospace;
        color: white;
    }
    #title {
        font-size: 25px;
        font-weight: 700;
        color: white;
        margin-bottom: 20px;
    }
    *[class=user-pass] {
        color: white;
        font-size: 15px;
        padding: 10px 8px;
        border: 1px solid #5f6368;
        border-radius: 6px;
    }
    *[class=user-pass]:focus {
        border: 1px solid #8ab4f8;
    }
    #login-button {
        color: black;
        font-size: 15px;
        font-weight: 600;
        padding: 10px;
        background-color: white;
        border-radius: 6px;
    }
    #login-button:hover {
        background-color: #dedede;
    }
    #error-label {
        color: #f28b82;
        font-size: 13px;
        font-weight: bold;
    }
"""

# Admin panel styles
ADMIN_STYLES = """
    #sidebar {
        background-color: #2a2b2e;
    }
    *[class=sidebar-buttons] {
        font-size: 16px;
        padding: 10px;
        border-radius: 6px;
        background-color: None;
        text-align: left;
        color: white;
        border: none;
    }
    *[class=sidebar-buttons]:checked {
        background-color: white;
        color: black;
    }
    *[class=sidebar-buttons]:hover:!checked {
        background-color: #3a3b3e;
    }
    *[class=start-button] {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 8px;
    }
    *[class=start-button]:hover {
        background-color: #45a049;
    }
    *[class=stop-button] {
        background-color: #f44336;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 8px;
    }
    *[class=stop-button]:hover {
        background-color: #da190b;
    }
    #camera-label {
        font-size: 14px;
        font-weight: 600;
    }
    #settingsCard {
        background-color: #2a2b2e;
        border-radius: 6px;
    }
    QFormLayout QLabel {
        font-size: 13px;
        background: none;
    }
"""

# Table styles
TABLE_STYLES = """
    QTableWidget {
        background-color: #2a2b2e;
        border: 1px solid #3a3b3e;
        border-radius: 6px;
        gridline-color: #3a3b3e;
        color: white;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QTableWidget::item:selected {
        background-color: #5f6368;
        color: white;
    }
    QHeaderView::section {
        background-color: #3a3b3e;
        color: white;
        padding: 5px;
        border: none;
        border-bottom: 1px solid #5f6368;
        font-weight: bold;
    }
    QTableCornerButton::section {
        background-color: #3a3b3e;
        border: none;
        border-bottom: 1px solid #5f6368;
    }
    QTableView::item:alternate {
        background-color: #313235;
    }
"""
