from typing import Dict, Any

# Verified agronomic & Indian seasonal metadata for the 22 dataset crops
CROP_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "rice": {
        "seasons": ["Kharif (June–November)", "Rabi (select irrigated zones)"],
        "water_requirement": "High (1,000–2,000 mm)",
        "soil_types": ["Clayey loam", "Alluvial", "Silty clay"],
        "major_indian_regions": ["Indo-Gangetic Plains", "Coastal Andhra & Telangana", "West Bengal", "Odisha", "Tamil Nadu"],
        "agronomic_note": "Requires standing water or assured irrigation during vegetative and reproductive stages."
    },
    "maize": {
        "seasons": ["Kharif (June–October)", "Rabi (October–March)"],
        "water_requirement": "Moderate (500–800 mm)",
        "soil_types": ["Well-drained sandy loam", "Alluvial", "Red loam"],
        "major_indian_regions": ["Karnataka", "Madhya Pradesh", "Telangana", "Bihar", "Maharashtra"],
        "agronomic_note": "Sensitive to waterlogging; requires well-drained neutral to slightly acidic soil."
    },
    "chickpea": {
        "seasons": ["Rabi (October–March)"],
        "water_requirement": "Low (300–450 mm)",
        "soil_types": ["Sandy loam", "Black cotton (Vertisols)"],
        "major_indian_regions": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Uttar Pradesh", "Karnataka"],
        "agronomic_note": "Fixes atmospheric nitrogen; thrives on residual soil moisture in cool, dry winters."
    },
    "kidneybeans": {
        "seasons": ["Kharif (Hills)", "Rabi (Plains)"],
        "water_requirement": "Moderate (400–600 mm)",
        "soil_types": ["Well-drained loam", "Clay loam"],
        "major_indian_regions": ["Himachal Pradesh", "Uttarakhand", "Jammu & Kashmir", "Maharashtra"],
        "agronomic_note": "Intolerant to frost and extreme heat during flowering; prefers mild temperatures."
    },
    "pigeonpeas": {
        "seasons": ["Kharif (June–December)"],
        "water_requirement": "Low to Moderate (600–650 mm)",
        "soil_types": ["Black cotton", "Red loam", "Alluvial"],
        "major_indian_regions": ["Maharashtra", "Karnataka", "Madhya Pradesh", "Telangana", "Gujarat"],
        "agronomic_note": "Deep-rooted drought-tolerant pulse; suitable for rainfed semi-arid tracts."
    },
    "mothbeans": {
        "seasons": ["Kharif (July–October)"],
        "water_requirement": "Very Low (250–400 mm)",
        "soil_types": ["Sandy", "Arid alluvial"],
        "major_indian_regions": ["Rajasthan", "Gujarat", "Haryana"],
        "agronomic_note": "Extremely heat- and drought-hardy arid legume; vulnerable to excess moisture."
    },
    "mungbean": {
        "seasons": ["Kharif (July–September)", "Zaid / Summer (March–June)"],
        "water_requirement": "Low (350–500 mm)",
        "soil_types": ["Well-drained loamy", "Sandy loam"],
        "major_indian_regions": ["Rajasthan", "Maharashtra", "Andhra Pradesh", "Telangana", "Odisha"],
        "agronomic_note": "Short-duration (60–70 days) nitrogen-fixing crop ideal for intercropping."
    },
    "blackgram": {
        "seasons": ["Kharif (June–September)", "Rabi (Coastal/South)"],
        "water_requirement": "Low to Moderate (400–600 mm)",
        "soil_types": ["Black cotton", "Loamy"],
        "major_indian_regions": ["Madhya Pradesh", "Andhra Pradesh", "Tamil Nadu", "Uttar Pradesh", "Maharashtra"],
        "agronomic_note": "Prefers warm, humid conditions with good drainage; sensitive to saline/alkaline soils."
    },
    "lentil": {
        "seasons": ["Rabi (October–March)"],
        "water_requirement": "Low (300–400 mm)",
        "soil_types": ["Alluvial", "Loamy", "Light black soils"],
        "major_indian_regions": ["Uttar Pradesh", "Madhya Pradesh", "Bihar", "West Bengal"],
        "agronomic_note": "Cool-season pulse tolerant to moderate drought; avoid poorly drained heavy clays."
    },
    "pomegranate": {
        "seasons": ["Perennial (Ambe, Mrig & Hasta Bahar)"],
        "water_requirement": "Low to Moderate (500–700 mm)",
        "soil_types": ["Light loamy", "Alluvial", "Medium black"],
        "major_indian_regions": ["Maharashtra", "Gujarat", "Karnataka", "Andhra Pradesh", "Telangana"],
        "agronomic_note": "Thrives in semi-arid climates with hot summers and cool winters; requires regulated irrigation."
    },
    "banana": {
        "seasons": ["Perennial / Year-round (planted June–July or Feb–March)"],
        "water_requirement": "Very High (1,200–2,200 mm)",
        "soil_types": ["Deep rich alluvial", "Fertile clay loam"],
        "major_indian_regions": ["Tamil Nadu", "Maharashtra", "Gujarat", "Andhra Pradesh", "Karnataka"],
        "agronomic_note": "Heavy feeder of potassium and nitrogen; requires frost-free humid climate and steady moisture."
    },
    "mango": {
        "seasons": ["Perennial (Fruiting: Summer / April–June)"],
        "water_requirement": "Moderate (750–1,000 mm)",
        "soil_types": ["Deep alluvial", "Lateritic", "Red loam"],
        "major_indian_regions": ["Uttar Pradesh", "Andhra Pradesh", "Telangana", "Karnataka", "Bihar", "Maharashtra"],
        "agronomic_note": "Requires dry weather during flowering and fruit maturity; rain during bloom reduces yield."
    },
    "grapes": {
        "seasons": ["Perennial (Pruning: April & October)"],
        "water_requirement": "Moderate (600–800 mm)",
        "soil_types": ["Well-drained sandy loam", "Light black soil"],
        "major_indian_regions": ["Maharashtra", "Karnataka", "Tamil Nadu", "Telangana"],
        "agronomic_note": "Dry, warm climate during berry ripening prevents fungal disease and boosts sugar content."
    },
    "watermelon": {
        "seasons": ["Zaid / Summer (January–May)"],
        "water_requirement": "Moderate (400–600 mm)",
        "soil_types": ["Sandy loam", "Riverbed alluvial"],
        "major_indian_regions": ["Uttar Pradesh", "Karnataka", "Andhra Pradesh", "Tamil Nadu", "Rajasthan"],
        "agronomic_note": "Warm-season cucurbit; high humidity and rain at maturity lower fruit sweetness."
    },
    "muskmelon": {
        "seasons": ["Zaid / Summer (February–May)"],
        "water_requirement": "Moderate (400–550 mm)",
        "soil_types": ["Sandy loam", "Well-drained alluvial"],
        "major_indian_regions": ["Uttar Pradesh", "Punjab", "Rajasthan", "Andhra Pradesh", "Madhya Pradesh"],
        "agronomic_note": "Needs hot, dry conditions and abundant sunshine during fruit ripening."
    },
    "apple": {
        "seasons": ["Perennial Temperate (Harvest: July–October)"],
        "water_requirement": "Moderate (1,000–1,200 mm)",
        "soil_types": ["Deep loamy", "Well-drained hill soils"],
        "major_indian_regions": ["Jammu & Kashmir", "Himachal Pradesh", "Uttarakhand"],
        "agronomic_note": "Requires 1,000–1,500 chilling hours below 7°C; unsuitable for tropical plains."
    },
    "orange": {
        "seasons": ["Perennial (Ambe & Mrig Bahar)"],
        "water_requirement": "Moderate (900–1,100 mm)",
        "soil_types": ["Well-drained alluvial", "Medium black", "Loamy"],
        "major_indian_regions": ["Maharashtra (Nagpur)", "Madhya Pradesh", "Punjab", "Rajasthan", "Assam"],
        "agronomic_note": "Sensitive to water stagnation and high soil salinity; requires well-aerated root zone."
    },
    "papaya": {
        "seasons": ["Perennial / Year-round (planted Feb–March or June–July)"],
        "water_requirement": "Moderate to High (1,000–1,500 mm)",
        "soil_types": ["Well-drained sandy loam", "Alluvial"],
        "major_indian_regions": ["Andhra Pradesh", "Gujarat", "Maharashtra", "Karnataka", "Madhya Pradesh"],
        "agronomic_note": "Strictly frost-sensitive and susceptible to collar rot if water stagnates."
    },
    "coconut": {
        "seasons": ["Perennial (Year-round harvest)"],
        "water_requirement": "High (1,300–2,000 mm)",
        "soil_types": ["Coastal sandy", "Alluvial", "Lateritic", "Red loam"],
        "major_indian_regions": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Coastal Odisha"],
        "agronomic_note": "Thrives in warm, humid coastal and tropical climates with high rainfall or irrigation."
    },
    "cotton": {
        "seasons": ["Kharif (May–November)"],
        "water_requirement": "Moderate (700–1,000 mm)",
        "soil_types": ["Deep black cotton (Regur)", "Alluvial"],
        "major_indian_regions": ["Gujarat", "Maharashtra", "Telangana", "Rajasthan", "Haryana"],
        "agronomic_note": "Requires frost-free growing period (180–200 days) and dry weather during boll opening."
    },
    "jute": {
        "seasons": ["Kharif / Pre-Kharif (March–August)"],
        "water_requirement": "High (1,200–1,500 mm)",
        "soil_types": ["New grey alluvial", "Silty loam"],
        "major_indian_regions": ["West Bengal", "Bihar", "Assam", "Odisha"],
        "agronomic_note": "Requires hot, humid climate (24–37°C, >70% humidity) and clean water for retting."
    },
    "coffee": {
        "seasons": ["Perennial Plantation (Harvest: November–February)"],
        "water_requirement": "High (1,500–2,500 mm)",
        "soil_types": ["Deep forest loam", "Lateritic", "Slightly acidic humic"],
        "major_indian_regions": ["Karnataka (Coorg/Chikmagalur)", "Kerala (Wayanad)", "Tamil Nadu (Nilgiris)"],
        "agronomic_note": "Shade-grown hill crop requiring well-distributed rainfall and slightly acidic soil (pH 6.0–6.5)."
    },
}


def get_regional_intelligence(crop: str, season_hint: str | None = None, region_hint: str | None = None) -> Dict[str, Any]:
    key = crop.strip().lower()
    info = CROP_KNOWLEDGE_BASE.get(key)
    if not info:
        return {
            "crop": crop,
            "verified_profile_available": False,
            "note": "No curated regional profile found; consult local KVK (Krishi Vigyan Kendra)."
        }

    season_match = None
    if season_hint:
        sh = season_hint.strip().lower()
        season_match = any(sh in s.lower() for s in info["seasons"]) or any("perennial" in s.lower() for s in info["seasons"])

    region_match = None
    if region_hint:
        rh = region_hint.strip().lower()
        region_match = any(rh in r.lower() for r in info["major_indian_regions"])

    return {
        "crop": key,
        "verified_profile_available": True,
        "recommended_seasons": info["seasons"],
        "water_requirement": info["water_requirement"],
        "preferred_soil_types": info["soil_types"],
        "major_indian_regions": info["major_indian_regions"],
        "agronomic_note": info["agronomic_note"],
        "user_context_alignment": {
            "provided_season": season_hint,
            "season_compatible": season_match,
            "provided_region": region_hint,
            "major_producing_region": region_match,
        }
    }