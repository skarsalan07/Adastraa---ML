import io
import sys
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest
import joblib

from .forms import UploadFileForm
from ml.preprocess import preprocess


def load_pipeline():
    model_path = Path(settings.BASE_DIR) / "ml" / "pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError("pipeline.pkl not found. Run training first.")
    return joblib.load(model_path)


def upload_and_predict(request):
    error_message = None
    form = UploadFileForm()

    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponseBadRequest("Invalid file upload")

        file = request.FILES["file"]

        try:
            df = pd.read_csv(file)
            original = df.copy()

            df_clean = preprocess(df)
            X = df_clean.drop(columns=["Sale_Amount"], errors="ignore")

            pipeline = load_pipeline()
            preds = pipeline.predict(X)

            original["Predicted_Sale_Amount"] = preds

            buffer = io.StringIO()
            original.to_csv(buffer, index=False)
            buffer.seek(0)

            response = HttpResponse(buffer.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = 'attachment; filename="predictions.csv"'
            return response

        except Exception as e:
            error_message = str(e)
            print("Error:", e, file=sys.stderr)

    return render(request, "prediction/upload.html", {"form": form, "error_message": error_message})
