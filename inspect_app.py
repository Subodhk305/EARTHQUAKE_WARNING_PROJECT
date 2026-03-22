import requests
import json

base_url = "http://localhost:8000"

def inspect_app():
    print("=" * 50)
    print("INSPECTING FASTAPI APP")
    print("=" * 50)
    
    # Get OpenAPI schema
    try:
        r = requests.get(f"{base_url}/openapi.json")
        print(f"✅ OpenAPI schema loaded")
        schema = r.json()
        print(f"API Title: {schema.get('info', {}).get('title', 'Unknown')}")
        print(f"API Version: {schema.get('info', {}).get('version', 'Unknown')}")
        
        # List all paths from schema
        paths = schema.get('paths', {})
        print(f"\nPaths in OpenAPI schema ({len(paths)}):")
        for path in paths.keys():
            print(f"  - {path}")
            
    except Exception as e:
        print(f"❌ Failed to get OpenAPI schema: {e}")

if __name__ == "__main__":
    inspect_app()