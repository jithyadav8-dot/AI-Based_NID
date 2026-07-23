import joblib, json
model = joblib.load("models/xgboost_nids.pkl")
with open("processed/feature_list.json") as f:
    features = json.load(f)

imp = dict(zip(features, model.feature_importances_))
protocol_imp = imp.get('Protocol', 'NOT IN MODEL')

if isinstance(protocol_imp, str):
    print(f"Protocol importance: {protocol_imp}")
else:
    print(f"Protocol importance: {protocol_imp:.4f}")
print(f"Total features: {len(features)}")
