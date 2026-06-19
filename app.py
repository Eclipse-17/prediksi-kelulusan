from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("random_forest_model.pkl")


def map_binary_feature(val):
    if pd.isna(val) or str(val).strip().lower() in [
        "tidak ada",
        "0",
        "none",
        "null",
        "",
        "false"
    ]:
        return 0
    return 1


@app.route("/")
def home():
    return jsonify({
        "message": "API Prediksi Risiko Keterlambatan Mahasiswa Aktif"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Gender
        gender = str(data["l/p"]).strip().upper()
        gender = 1 if gender == "L" else 0

        # IPK
        ipk = float(data["ipk"])

        # Prestasi
        prestasi = map_binary_feature(data.get("prestasi", 0))

        # Jurnal
        jurnal = map_binary_feature(data.get("jurnal", 0))

        # Susun sesuai urutan saat training
        input_data = pd.DataFrame([{
            "l/p": gender,
            "ipk": ipk,
            "prestasi": prestasi,
            "jurnal": jurnal
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        return jsonify({
            "status": "success",
            "prediksi": "Terlambat" if prediction == 1 else "Tepat Waktu",
            "kode_prediksi": int(prediction),
            "probabilitas_tepat_waktu": round(float(probability[0]), 4),
            "probabilitas_terlambat": round(float(probability[1]), 4)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == "__main__":
    app.run()