# backend/db_init.py
import os
import duckdb

os.makedirs("backend/data", exist_ok=True)
db_path = "backend/data/analytics.duckdb"
con = duckdb.connect(db_path)

con.execute("DROP TABLE IF EXISTS sales_messy;")

con.execute("""
CREATE TABLE sales_messy (
    order_id INTEGER,
    \"2020\" VARCHAR,
    amount_str VARCHAR,
    prod_cat VARCHAR,
    created_on VARCHAR
);
""")

con.execute("""
INSERT INTO sales_messy VALUES
 (1, '2024-01-05', '1,200.50', 'Electronics', '2024-01-05'),
 (2, '2024-02-10', '2,400', 'Electronics', '10-02-2024'),
 (3, '2024-02-28', '350', 'Furniture', '2024/02/28'),
 (4, '2024-03-01', '5,000', 'Electronics', '01-Mar-2024'),
 (5, '2024-03-15', '7,200.75', 'Furniture', '2024-03-15'),
 (6, '2024-04-01', '0', 'Services', '2024-04-01');
""")

con.commit()
print("Seeded backend/data/analytics.duckdb with 'sales_messy'.")
