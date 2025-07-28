from database.db_setup import init_db, SessionLocal
from database.db_models import InstanceDB

# 1. Tabelle erzeugen, falls noch nicht vorhanden
init_db()

# 2. Session starten
session = SessionLocal()

# 3. Neue Instanz anlegen und speichern
new_inst = InstanceDB(name="Maschine X")
session.add(new_inst)
session.commit()

# 4. Instanz abrufen
instance_from_db = session.query(InstanceDB).first()

# 5. Ausgabe als JSON
print(instance_from_db)