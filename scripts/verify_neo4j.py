from src.graph.neo4j_manager import Neo4jManager
import os

def main():
    manager = Neo4jManager()
    with manager.driver.session() as session:
        print("--- Node Counts ---")
        result = session.run("MATCH (n) RETURN labels(n) as label, count(n) as count")
        for record in result:
            print(f"{record['label']}: {record['count']}")
        
        print("\n--- Relationship Counts ---")
        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
        for record in result:
            print(f"{record['type']}: {record['count']}")

        print("\n--- Sample Document ---")
        result = session.run("MATCH (d:Document) RETURN d LIMIT 1")
        for record in result:
            print(record['d'])

        print("\n--- Sample Question ---")
        result = session.run("MATCH (q:Question) RETURN q LIMIT 1")
        for record in result:
            print(record['q'])

    manager.close()

if __name__ == "__main__":
    main()
