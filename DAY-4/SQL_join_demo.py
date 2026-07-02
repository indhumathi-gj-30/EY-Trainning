import pandas as pd
import sqlite3

# due to inner join SRV-03 is missing
# Create in-memory SQLite database
database_conn = sqlite3.connect(':memory:')

# Data setup
server_data = {
    "Host_ID": ["SRV-01", "SRV-02", "SRV-03"],
    "Role": ["Web Front", "API Gateway", "Database Replica"]
}

interface_data = {
    "Interface_ID": ["eth0", "eth1"],
    "Mapped_Host": ["SRV-01", "SRV-02"],
    "IP_Address": ["10.0.0.4", "10.0.0.9"]
}

# Convert to DataFrames
servers_df = pd.DataFrame(server_data)
interfaces_df = pd.DataFrame(interface_data)

# Load into SQL tables
servers_df.to_sql('Servers', database_conn, index=False, if_exists='replace')
interfaces_df.to_sql('Interfaces', database_conn, index=False, if_exists='replace')

faulty_join_query = """
SELECT s.Host_ID, s.Role, i.Interface_ID, i.IP_Address
FROM Servers s
INNER JOIN Interfaces i
ON s.Host_ID = i.Mapped_Host;
"""

print("\n-INNER JOIN Result (SRV-03 Missing)")
inner_join_result = pd.read_sql_query(faulty_join_query, database_conn)
print(inner_join_result)

correct_join_query = """
SELECT s.Host_ID, s.Role, i.Interface_ID, i.IP_Address
FROM Servers s
LEFT JOIN Interfaces i
ON s.Host_ID = i.Mapped_Host;
"""

print("\n--- LEFT JOIN Result (All Servers Included) ---")
left_join_result = pd.read_sql_query(correct_join_query, database_conn)
print(left_join_result)

# Close connection
database_conn.close()