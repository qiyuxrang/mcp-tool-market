"""Initialize the sample SQLite database for the DB MCP Server."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)

    # Create orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer TEXT,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            status TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Insert sample products
    products = [
        (1, "Laptop Pro 16", "Electronics", 12999.0, 15),
        (2, "Wireless Mouse", "Electronics", 299.0, 50),
        (3, "Mechanical Keyboard", "Electronics", 899.0, 30),
        (4, "Desk Lamp", "Furniture", 259.0, 25),
        (5, "Ergonomic Chair", "Furniture", 3299.0, 10),
        (6, "Standing Desk", "Furniture", 4599.0, 8),
        (7, "Notebook A5", "Stationery", 29.0, 200),
        (8, "Ballpoint Pen Pack", "Stationery", 49.0, 150),
        (9, "USB-C Hub", "Electronics", 399.0, 40),
        (10, "Monitor Stand", "Furniture", 599.0, 20),
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO products (id, name, category, price, stock) VALUES (?, ?, ?, ?, ?)",
        products,
    )

    # Insert sample orders
    orders = [
        (1, "Alice", 1, 1, "2026-06-01", "delivered"),
        (2, "Alice", 2, 2, "2026-06-01", "delivered"),
        (3, "Bob", 3, 1, "2026-06-15", "shipped"),
        (4, "Charlie", 5, 1, "2026-06-20", "processing"),
        (5, "Diana", 7, 10, "2026-06-25", "pending"),
        (6, "Eve", 9, 2, "2026-06-28", "pending"),
        (7, "Bob", 4, 1, "2026-07-01", "processing"),
        (8, "Charlie", 1, 1, "2026-07-02", "pending"),
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO orders (id, customer, product_id, quantity, order_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        orders,
    )

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print(f"  - {len(products)} products inserted")
    print(f"  - {len(orders)} orders inserted")


if __name__ == "__main__":
    create_database()
