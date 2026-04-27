from src.graph.neo4j_manager import Neo4jManager
from dotenv import load_dotenv
import os

load_dotenv()

def test_connection():
    try:
        manager = Neo4jManager()
        
        # Setup constraints and indexes
        manager.setup_database()
        
        with manager.driver.session() as session:
            result = session.run("RETURN 1")
            print("Successfully connected to Neo4j.")
            
            # Print current counts
            counts = session.run("""
                MATCH (n)
                RETURN labels(n) as label, count(*) as count
            """)
            print("Current Database Stats:")
            for record in counts:
                print(f"Label: {record['label']}, Count: {record['count']}")
        manager.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

if __name__ == "__main__":
    test_connection()
