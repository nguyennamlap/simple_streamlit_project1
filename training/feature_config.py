

# Target variable
TARGET = 'TARGET'

# Identifier
ID_COLUMN = 'SK_ID_CURR'

# ============================================================================
# CATEGORICAL FEATURES
# ============================================================================

CATEGORICAL_FEATURES = {
    'contract_type': ['NAME_CONTRACT_TYPE'],
    
    'demographic': [
        'CODE_GENDER',
        'NAME_FAMILY_STATUS',
        'NAME_EDUCATION_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_HOUSING_TYPE',
    ],
    
    'application': [
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START',
    ],
    
    'building': [
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE',
    ],
    
    'engineered': [
        'Sở hữu xe',
        'OCCUPATION_TYPE_ENHANCED',
        'OCCUPATION_MISSING_TYPE',
    ]
}

# Flatten all categorical features
ALL_CATEGORICAL = [feat for group in CATEGORICAL_FEATURES.values() for feat in group]

# ============================================================================
# NUMERICAL FEATURES
# ============================================================================

NUMERICAL_FEATURES = {
    'amounts': [
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
    ],
    
    'counts': [
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
    ],
    
    'time_features': [
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE',
        'OWN_CAR_AGE',
    ],
    
    'external_sources': [
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
    ],
    
    'region': [
        'REGION_POPULATION_RELATIVE',
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY',
    ],
    
    'building_avg': [
        'APARTMENTS_AVG',
        'BASEMENTAREA_AVG',
        'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG',
        'COMMONAREA_AVG',
        'ELEVATORS_AVG',
        'ENTRANCES_AVG',
        'FLOORSMAX_AVG',
        'FLOORSMIN_AVG',
        'LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG',
    ],
    
    'building_mode': [
        'APARTMENTS_MODE',
        'BASEMENTAREA_MODE',
        'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE',
        'COMMONAREA_MODE',
        'ELEVATORS_MODE',
        'ENTRANCES_MODE',
        'FLOORSMAX_MODE',
        'FLOORSMIN_MODE',
        'LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE',
        'TOTALAREA_MODE',
    ],
    
    'building_medi': [
        'APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI',
        'COMMONAREA_MEDI',
        'ELEVATORS_MEDI',
        'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI',
        'LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI',
    ],
    
    'application_timing': [
        'HOUR_APPR_PROCESS_START',
    ],
    
    'credit_bureau': [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
    ],
    
    'engineered': [
        'Tỉ lệ vay so với nhu cầu',
    ]
}

# Flatten all numerical features
ALL_NUMERICAL = [feat for group in NUMERICAL_FEATURES.values() for feat in group]

# ============================================================================
# BINARY FEATURES
# ============================================================================

BINARY_FEATURES = {
    'ownership': [
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
    ],
    
    'contact': [
        'FLAG_MOBIL',
        'FLAG_EMP_PHONE',
        'FLAG_WORK_PHONE',
        'FLAG_CONT_MOBILE',
        'FLAG_PHONE',
        'FLAG_EMAIL',
    ],
    
    'region_flags': [
        'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY',
    ],
    
    'documents': [
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_3',
        'FLAG_DOCUMENT_4',
        'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6',
        'FLAG_DOCUMENT_7',
        'FLAG_DOCUMENT_8',
        'FLAG_DOCUMENT_9',
        'FLAG_DOCUMENT_10',
        'FLAG_DOCUMENT_11',
        'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13',
        'FLAG_DOCUMENT_14',
        'FLAG_DOCUMENT_15',
        'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_18',
        'FLAG_DOCUMENT_19',
        'FLAG_DOCUMENT_20',
        'FLAG_DOCUMENT_21',
    ],
    
    'missing_indicators': [
        'EXT_SOURCE_1_is_missing',
        'EXT_SOURCE_2_is_missing',
        'EXT_SOURCE_3_is_missing',
    ],
    
    'engineered': [
        'IS_RETIRED_NO_OCCUPATION',
        'IS_WORKING_NO_OCCUPATION',
    ]
}

# Flatten all binary features
ALL_BINARY = [feat for group in BINARY_FEATURES.values() for feat in group]

# ============================================================================
# FEATURE GROUPS FOR SPECIAL HANDLING
# ============================================================================

# Features with high missing values (may need special imputation)
HIGH_MISSING_FEATURES = [
    'OWN_CAR_AGE',
    'OCCUPATION_TYPE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
] + NUMERICAL_FEATURES['building_avg'] + NUMERICAL_FEATURES['building_mode'] + NUMERICAL_FEATURES['building_medi']

# Features for log transformation (highly skewed)
LOG_TRANSFORM_FEATURES = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
]

# Features for scaling (different ranges)
SCALE_FEATURES = ALL_NUMERICAL

# External source features (important predictors)
EXTERNAL_SOURCES = [
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
]

# Time-based features (need conversion to positive values)
TIME_FEATURES = [
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH',
    'DAYS_LAST_PHONE_CHANGE',
]

# Document flags for aggregation
DOCUMENT_FLAGS = BINARY_FEATURES['documents']

# Building features (can be aggregated)
BUILDING_FEATURES = (
    NUMERICAL_FEATURES['building_avg'] + 
    NUMERICAL_FEATURES['building_mode'] + 
    NUMERICAL_FEATURES['building_medi'] +
    CATEGORICAL_FEATURES['building']
)

# ============================================================================
# FEATURE IMPORTANCE GROUPS
# ============================================================================

# Core features (most important based on domain knowledge)
CORE_FEATURES = [
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'CODE_GENDER',
    'NAME_EDUCATION_TYPE',
    'NAME_INCOME_TYPE',
    'REGION_RATING_CLIENT',
]

# Engineered features
ENGINEERED_FEATURES = [
    'Tỉ lệ vay so với nhu cầu',
    'Sở hữu xe',
    'OCCUPATION_TYPE_ENHANCED',
    'OCCUPATION_MISSING_TYPE',
    'IS_RETIRED_NO_OCCUPATION',
    'IS_WORKING_NO_OCCUPATION',
    'EXT_SOURCE_1_is_missing',
    'EXT_SOURCE_2_is_missing',
    'EXT_SOURCE_3_is_missing',
]

# ============================================================================
# ALL FEATURES
# ============================================================================

ALL_FEATURES = ALL_CATEGORICAL + ALL_NUMERICAL + ALL_BINARY

# Exclude ID and Target from feature list
MODELING_FEATURES = [f for f in ALL_FEATURES if f not in [ID_COLUMN, TARGET]]

# ============================================================================
# PREPROCESSING CONFIGURATIONS
# ============================================================================

PREPROCESSING_CONFIG = {
    'categorical': {
        'strategy': 'mode',  # or 'constant'
        'encoder': 'onehot',  # or 'label', 'target'
        'handle_unknown': 'ignore',
    },
    
    'numerical': {
        'strategy': 'median',  # or 'mean', 'constant'
        'scaler': 'standard',  # or 'minmax', 'robust'
    },
    
    'binary': {
        'strategy': 'constant',
        'fill_value': 0,
    },
    
    'time_features': {
        'convert_to_positive': True,
        'unit': 'years',  # Convert days to years
    },
    
    'external_sources': {
        'create_missing_indicators': True,
        'imputation': 'median',
        'create_interactions': True,
    }
}

# ============================================================================
# FEATURE ENGINEERING SUGGESTIONS
# ============================================================================

FEATURE_ENGINEERING = {
    'ratios': [
        ('CREDIT_INCOME_RATIO', 'AMT_CREDIT', 'AMT_INCOME_TOTAL'),
        ('ANNUITY_INCOME_RATIO', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
        ('CREDIT_GOODS_RATIO', 'AMT_CREDIT', 'AMT_GOODS_PRICE'),
        ('ANNUITY_CREDIT_RATIO', 'AMT_ANNUITY', 'AMT_CREDIT'),
    ],
    
    'aggregations': [
        ('TOTAL_DOCUMENTS', DOCUMENT_FLAGS, 'sum'),
        ('EXT_SOURCE_MEAN', EXTERNAL_SOURCES, 'mean'),
        ('EXT_SOURCE_MAX', EXTERNAL_SOURCES, 'max'),
        ('EXT_SOURCE_MIN', EXTERNAL_SOURCES, 'min'),
    ],
    
    'transformations': [
        ('AGE_YEARS', 'DAYS_BIRTH', lambda x: abs(x) / 365),
        ('EMPLOYMENT_YEARS', 'DAYS_EMPLOYED', lambda x: abs(x) / 365),
        ('INCOME_PER_PERSON', 'AMT_INCOME_TOTAL', lambda df: df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']),
    ],
    
    'interactions': [
        ('EXT_SOURCE_1_2', ['EXT_SOURCE_1', 'EXT_SOURCE_2'], 'multiply'),
        ('EXT_SOURCE_1_3', ['EXT_SOURCE_1', 'EXT_SOURCE_3'], 'multiply'),
        ('EXT_SOURCE_2_3', ['EXT_SOURCE_2', 'EXT_SOURCE_3'], 'multiply'),
    ]
}

# ============================================================================
# FEATURE DESCRIPTIONS
# ============================================================================

FEATURE_DESCRIPTIONS = {
    'SK_ID_CURR': 'Loan application ID',
    'TARGET': 'Target variable (1 = client with payment difficulties, 0 = all other cases)',
    'NAME_CONTRACT_TYPE': 'Type of loan contract',
    'CODE_GENDER': 'Gender of the client',
    'FLAG_OWN_CAR': 'Flag if the client owns a car',
    'FLAG_OWN_REALTY': 'Flag if client owns a house or flat',
    'CNT_CHILDREN': 'Number of children the client has',
    'AMT_INCOME_TOTAL': 'Income of the client',
    'AMT_CREDIT': 'Credit amount of the loan',
    'AMT_ANNUITY': 'Loan annuity',
    'AMT_GOODS_PRICE': 'For consumer loans it is the price of the goods for which the loan is given',
    'EXT_SOURCE_1': 'Normalized score from external data source 1',
    'EXT_SOURCE_2': 'Normalized score from external data source 2',
    'EXT_SOURCE_3': 'Normalized score from external data source 3',
    'DAYS_BIRTH': 'Client\'s age in days at the time of application',
    'DAYS_EMPLOYED': 'How many days before the application the person started current employment',
    'Tỉ lệ vay so với nhu cầu': 'Loan to goods price ratio (engineered)',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_group(feature_name):
    """Get the group name for a given feature"""
    for group_type in [CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES]:
        for group_name, features in group_type.items():
            if feature_name in features:
                return group_name
    return 'unknown'

def get_feature_type(feature_name):
    """Get the type (categorical/numerical/binary) for a given feature"""
    if feature_name in ALL_CATEGORICAL:
        return 'categorical'
    elif feature_name in ALL_NUMERICAL:
        return 'numerical'
    elif feature_name in ALL_BINARY:
        return 'binary'
    return 'unknown'

def print_feature_summary():
    """Print a summary of all feature groups"""
    print(f"Total Features: {len(ALL_FEATURES)}")
    print(f"Categorical Features: {len(ALL_CATEGORICAL)}")
    print(f"Numerical Features: {len(ALL_NUMERICAL)}")
    print(f"Binary Features: {len(ALL_BINARY)}")
    print(f"Engineered Features: {len(ENGINEERED_FEATURES)}")
    print(f"\nModeling Features (excluding ID and Target): {len(MODELING_FEATURES)}")

if __name__ == "__main__":
    print_feature_summary()