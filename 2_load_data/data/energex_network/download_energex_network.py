"""Download Energex network layers (zone subs, distribution subs, 11kV lines/cables) from the public ArcGIS feature service behind https://www.arcgis.com/home/item.html?id=49d914e6bc3143029e690c0a01b072cf"""
import json, os, time, requests

URL = "https://services.arcgis.com/bfVzktoY0OhzQCDj/arcgis/rest/services/Network_Energex/FeatureServer"
LAYERS = {
    0:  "zone_substations.geojson",
    1:  "distribution_substations.geojson",
    10: "hv_11kV_lines.geojson",
    11: "hv_11kV_cables.geojson",
}
PAGE = 1000
HERE = os.path.dirname(os.path.abspath(__file__))

def get(params, tries=4):
    path = params.pop("_layer")
    for t in range(tries):
        try:
            r = requests.get(URL + path, params=params, timeout=120)
            r.raise_for_status(); return r.json()
        except Exception as e:
            if t == tries - 1: raise
            print("   retry:", e); time.sleep(5 * (t + 1))

for lid, fname in LAYERS.items():
    n = get({"_layer": f"/{lid}/query", "where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    print(f"Layer {lid} -> {fname}: {n} features")
    feats, offset = [], 0
    while offset < n:
        js = get({"_layer": f"/{lid}/query", "where": "1=1", "outFields": "*", "outSR": 4326,
                  "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": PAGE,
                  "f": "geojson"})
        page = js.get("features", [])
        if not page: break
        feats += page; offset += len(page)
        print(f"   {offset}/{n}", end="\r")
    print(f"   got {len(feats)} of {n}" + ("" if len(feats) == n else "  <-- MISMATCH, re-run"))
    with open(os.path.join(HERE, fname), "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
print("Done.")
