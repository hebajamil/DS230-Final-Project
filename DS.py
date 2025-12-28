import pandas as pd 
import numpy as np
from pathlib import Path

DATA_DIR=Path("/content/drive/Mydrive/data")# Set the base directory for all dataset files to ensure organized file access
print(DATA_DIR)

dtypes_orders = {
    "order_id": np.int32,
    "user_id": np.int32,
    "order_number": np.int16,
    "order_dow": np.int8,
    "order_hour_of_day": np.int8,
    "days_since_prior_order": np.float32,  # has NaNs
    "eval_set": "category"
}

dtypes_op = {
    "order_id": np.int32,
    "product_id": np.int32,
    "add_to_cart_order": np.int16,
    "reordered": np.int8
}

dtypes_products = {
    "product_id": np.int32,
    "product_name": "category",
    "aisle_id": np.int16,
    "department_id": np.int16
}

dtypes_aisles = {
    "aisle_id": np.int16,
    "aisle": "category"
}

dtypes_departments = {
    "department_id": np.int16,
    "department": "category"
}
orders = pd.read_csv(DATA_DIR / "orders.csv", dtype=dtypes_orders)
op_prior = pd.read_csv(DATA_DIR / "order_products__prior.csv", dtype=dtypes_op)
op_train = pd.read_csv(DATA_DIR / "order_products__train.csv", dtype=dtypes_op)
products = pd.read_csv(DATA_DIR / "products.csv", dtype=dtypes_products)
aisles = pd.read_csv(DATA_DIR / "aisles.csv", dtype=dtypes_aisles)
departments = pd.read_csv(DATA_DIR / "departments.csv", dtype=dtypes_departments)

orders.head()
op_prior.head()
op_train.head()
products.head()
aisles.head()
departments.head()