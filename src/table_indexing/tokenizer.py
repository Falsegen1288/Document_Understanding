import re
from typing import List, Tuple, Optional, Any, Dict

class TableEntityTokenizer:
    """
    Custom BM25 regex tokenizer preserving exact part/catalog numbers, SKUs,
    and technical parameters (e.g., SKU-4471, CAT-9021-B, 5.0V, >=10mm).
    Reuses existing BM25 regex logic for structured lookups.
    """
    
    # Regex pattern to capture alphanumeric identifiers, model numbers, hyphenated SKUs, and units
    ENTITY_TOKEN_PATTERN = re.compile(
        r'[A-Za-z0-9]+(?:[-_./:][A-Za-z0-9]+)*'
    )
    
    # Range / predicate regex pattern (e.g. >= 10 mm, <= 40°C, > 5)
    PREDICATE_PATTERN = re.compile(
        r'([><]=?|=)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z°μµ%]+)?'
    )
    
    # Unit equivalences for unit normalization (base unit scaling)
    UNIT_SCALES = {
        'v': ('voltage', 1.0),
        'volt': ('voltage', 1.0),
        'volts': ('voltage', 1.0),
        'mv': ('voltage', 1e-3),
        'kv': ('voltage', 1e3),
        
        'mm': ('length', 1e-3),
        'cm': ('length', 1e-2),
        'm': ('length', 1.0),
        'inch': ('length', 0.0254),
        'in': ('length', 0.0254),
        
        '°c': ('temp', 1.0),
        'c': ('temp', 1.0),
        'degc': ('temp', 1.0),
        'k': ('temp', 1.0),
        
        'w': ('power', 1.0),
        'mw': ('power', 1e-3),
        'kw': ('power', 1e3),
        
        'hz': ('freq', 1.0),
        'khz': ('freq', 1e3),
        'mhz': ('freq', 1e6),
        'ghz': ('freq', 1e9),
        
        'gb': ('memory', 1.0),
        'tb': ('memory', 1024.0),
        'mb': ('memory', 1/1024.0),
    }

    ABBREVIATIONS = {
        'temp': 'temperature',
        'temperature': 'temperature',
        'max': 'maximum',
        'maximum': 'maximum',
        'min': 'minimum',
        'minimum': 'minimum',
        'volts': 'voltage',
        'volt': 'voltage',
        'v': 'voltage',
        'voltage': 'voltage',
        'dia': 'diameter',
        'diameter': 'diameter',
        'spec': 'specification',
        'specs': 'specification',
        'capac': 'capacity',
        'capacity': 'capacity',
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Extract preserved entity tokens from query or text string with abbreviation expansion."""
        if not text:
            return []
        tokens = cls.ENTITY_TOKEN_PATTERN.findall(text.lower())
        expanded = []
        for t in tokens:
            expanded.append(t)
            if t in cls.ABBREVIATIONS:
                expanded.append(cls.ABBREVIATIONS[t])
        return list(set(expanded))

    @classmethod
    def extract_part_numbers(cls, text: str) -> List[str]:
        """Extract explicit part numbers or SKU-like codes."""
        tokens = cls.tokenize(text)
        # Filter tokens containing hyphens/numbers/letters combination typical of SKUs
        sku_candidates = [
            t for t in tokens 
            if re.search(r'[0-9]', t) and (re.search(r'[a-z]', t) or '-' in t or '_' in t or '.' in t)
        ]
        return sku_candidates

    @classmethod
    def normalize_unit_value(cls, text: str) -> Tuple[Optional[float], Optional[str], str]:
        """
        Parse a string value into (numeric_val, canonical_unit_category, normalized_string).
        Handles variants: '5V', '5.0V', '5 volts', '5000 mV'.
        """
        text_clean = text.strip()
        match = re.search(r'^([+-]?[0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z°μµ%]+)?$', text_clean)
        if not match:
            return None, None, text_clean.lower()
            
        val_str, unit_str = match.groups()
        val = float(val_str)
        if not unit_str:
            return val, 'dimensionless', f"{val:.4g}"
            
        unit_lower = unit_str.lower()
        if unit_lower in cls.UNIT_SCALES:
            category, scale = cls.UNIT_SCALES[unit_lower]
            base_val = val * scale
            return base_val, category, f"{val:.4g} {unit_lower}"
            
        return val, unit_lower, f"{val:.4g} {unit_lower}"

    @classmethod
    def parse_predicate(cls, query: str) -> Optional[Dict[str, Any]]:
        """Extract filter predicate from query string like 'max operating voltage >= 10V'."""
        match = cls.PREDICATE_PATTERN.search(query)
        if match:
            op, val_str, unit_str = match.groups()
            val = float(val_str)
            unit_cat = None
            if unit_str and unit_str.lower() in cls.UNIT_SCALES:
                unit_cat, scale = cls.UNIT_SCALES[unit_str.lower()]
                val = val * scale
            return {
                'operator': op,
                'target_val': val,
                'unit_category': unit_cat,
                'raw_unit': unit_str
            }
        return None
