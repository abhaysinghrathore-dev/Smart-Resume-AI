import sys
import os

# Add the current directory to sys.path so we can import config
sys.path.append(os.getcwd())

from config.database import init_database, add_admin

print("Initializing database...")
init_database()
print("Database initialized.")

# Get credentials from environment variables or use defaults (for dev only)
admin_email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')

print("Adding default admin...")
success = add_admin(admin_email, admin_password)
if success:
    print(f"Default admin added: {admin_email}")
else:
    print("Failed to add admin (might already exist).")
