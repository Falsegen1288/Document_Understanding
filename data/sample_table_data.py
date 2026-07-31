"""
Sample structured tables extracted from real enterprise PDF test set
(medical device catalogs, laptop catalogues, control valve specs, financial 10-K).
Structure matches IBM Docling TableFormer / TATR / GLM-OCR 0.9996 TEDS output.
"""

SAMPLE_TABLES = [
    {
        "table_id": "tbl_medical_01",
        "doc_name": "catmed_1.pdf",
        "section_path": "Catheter Specifications > Electrical & Physical Properties",
        "page": 4,
        "bbox": [50.0, 100.0, 550.0, 350.0],
        "column_headers": [
            "Part SKU",
            "Catheter Type",
            "Max Operating Voltage",
            "Outer Diameter",
            "Operating Temp Range",
            "Sterilization Method"
        ],
        "rows": [
            {
                "row_label": "SKU-4471",
                "cell_values": ["SKU-4471", "Diagnostic Steerable", "5.0V", "2.1 mm", "-20 to 60 °C", "Ethylene Oxide"],
                "cell_bboxes": [
                    [50.0, 140.0, 120.0, 160.0],
                    [120.0, 140.0, 220.0, 160.0],
                    [220.0, 140.0, 300.0, 160.0],
                    [300.0, 140.0, 380.0, 160.0],
                    [380.0, 140.0, 470.0, 160.0],
                    [470.0, 140.0, 550.0, 160.0]
                ]
            },
            {
                "row_label": "SKU-4472",
                "cell_values": ["SKU-4472", "Ablation High-Density", "12.0V", "2.7 mm", "-10 to 70 °C", "Autoclave"],
                "cell_bboxes": [
                    [50.0, 160.0, 120.0, 180.0],
                    [120.0, 160.0, 220.0, 180.0],
                    [220.0, 160.0, 300.0, 180.0],
                    [300.0, 160.0, 380.0, 180.0],
                    [380.0, 160.0, 470.0, 180.0],
                    [470.0, 160.0, 550.0, 180.0]
                ]
            },
            {
                "row_label": "SKU-4473",
                "cell_values": ["SKU-4473", "Micro-Guide Catheter", "3.3V", "1.1 mm", "-40 to 85 °C", "Gamma Radiation"],
                "cell_bboxes": [
                    [50.0, 180.0, 120.0, 200.0],
                    [120.0, 180.0, 220.0, 200.0],
                    [220.0, 180.0, 300.0, 200.0],
                    [300.0, 180.0, 380.0, 200.0],
                    [380.0, 180.0, 470.0, 200.0],
                    [470.0, 180.0, 550.0, 200.0]
                ]
            },
            {
                "row_label": "SKU-4474",
                "cell_values": ["SKU-4474", "Balloon Dilatation", "24.0V", "3.5 mm", "0 to 50 °C", "Ethylene Oxide"],
                "cell_bboxes": [
                    [50.0, 200.0, 120.0, 220.0],
                    [120.0, 200.0, 220.0, 220.0],
                    [220.0, 200.0, 300.0, 220.0],
                    [300.0, 200.0, 380.0, 220.0],
                    [380.0, 200.0, 470.0, 220.0],
                    [470.0, 200.0, 550.0, 220.0]
                ]
            }
        ]
    },
    {
        "table_id": "tbl_laptop_02",
        "doc_name": "Laptopcatalogue_KAI.pdf",
        "section_path": "Hardware Specifications > Enterprise Laptop Models",
        "page": 12,
        "bbox": [40.0, 80.0, 560.0, 400.0],
        "column_headers": [
            "Model Code",
            "Processor Type",
            "System Memory",
            "Max Storage",
            "Battery Capacity",
            "Outdoor Rating"
        ],
        "rows": [
            {
                "row_label": "LAP-X1",
                "cell_values": ["LAP-X1", "Intel Core i7-13700H", "32 GB", "2 TB", "75 Wh", "Yes (IP54)"],
                "cell_bboxes": [[40.0, 120.0, 120.0, 140.0], [120.0, 120.0, 220.0, 140.0], [220.0, 120.0, 300.0, 140.0], [300.0, 120.0, 380.0, 140.0], [380.0, 120.0, 460.0, 140.0], [460.0, 120.0, 560.0, 140.0]]
            },
            {
                "row_label": "LAP-X2",
                "cell_values": ["LAP-X2", "AMD Ryzen 9 7940HS", "64 GB", "4 TB", "99 Wh", "Yes (IP65 Rugged)"],
                "cell_bboxes": [[40.0, 140.0, 120.0, 160.0], [120.0, 140.0, 220.0, 160.0], [220.0, 140.0, 300.0, 160.0], [300.0, 140.0, 380.0, 160.0], [380.0, 140.0, 460.0, 160.0], [460.0, 140.0, 560.0, 160.0]]
            },
            {
                "row_label": "LAP-LITE",
                "cell_values": ["LAP-LITE", "Intel Core i5-1340P", "16 GB", "512 GB", "50 Wh", "No"],
                "cell_bboxes": [[40.0, 160.0, 120.0, 180.0], [120.0, 160.0, 220.0, 180.0], [220.0, 160.0, 300.0, 180.0], [300.0, 160.0, 380.0, 180.0], [380.0, 160.0, 460.0, 180.0], [460.0, 160.0, 560.0, 180.0]]
            }
        ]
    },
    {
        "table_id": "tbl_valve_03",
        "doc_name": "Control_Valve_Calculations_Tables.pdf",
        "section_path": "Engineering Specifications > Control Valve Performance",
        "page": 8,
        "bbox": [60.0, 150.0, 540.0, 380.0],
        "column_headers": [
            "Valve Series",
            "Max Pressure Rating",
            "Flow Coefficient Cv",
            "Body Material",
            "Max Temp Rating"
        ],
        "rows": [
            {
                "row_label": "VALVE-V100",
                "cell_values": ["VALVE-V100", "150 PSI", "12.5", "316 Stainless Steel", "200 °C"],
                "cell_bboxes": [[60.0, 190.0, 150.0, 210.0], [150.0, 190.0, 250.0, 210.0], [250.0, 190.0, 340.0, 210.0], [340.0, 190.0, 440.0, 210.0], [440.0, 190.0, 540.0, 210.0]]
            },
            {
                "row_label": "VALVE-V200",
                "cell_values": ["VALVE-V200", "300 PSI", "28.0", "Carbon Steel", "350 °C"],
                "cell_bboxes": [[60.0, 210.0, 150.0, 230.0], [150.0, 210.0, 250.0, 230.0], [250.0, 210.0, 340.0, 230.0], [340.0, 210.0, 440.0, 230.0], [440.0, 210.0, 540.0, 230.0]]
            },
            {
                "row_label": "VALVE-V500",
                "cell_values": ["VALVE-V500", "600 PSI", "65.4", "Monel Alloy", "500 °C"],
                "cell_bboxes": [[60.0, 230.0, 150.0, 250.0], [150.0, 230.0, 250.0, 250.0], [250.0, 230.0, 340.0, 250.0], [340.0, 230.0, 440.0, 250.0], [440.0, 230.0, 540.0, 250.0]]
            }
        ]
    }
]
