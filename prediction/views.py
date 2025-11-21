import io
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

from ml.preprocess import preprocess
from .forms import UploadFileForm


def fig_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(img_png).decode("utf-8")


def load_pipeline():
    model_path = Path(settings.BASE_DIR) / "ml" / "pipeline.pkl"
    return joblib.load(model_path)


def upload_and_visualize(request):
    form = UploadFileForm()

    train_df = pd.read_csv(Path(settings.BASE_DIR) / "data" / "train.csv")
    train_df = preprocess(train_df)
    numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    plot_image = None
    error_message = None
    prediction_error = None

    if request.method == "GET" and ("feature" in request.GET):
        feature = request.GET.get("feature")
        plot_type = request.GET.get("plot_type")

        try:
            plt.figure(figsize=(7, 5))

            if plot_type == "scatter":
                sns.scatterplot(x=train_df[feature], y=train_df["Sale_Amount"])
                plt.xlabel(feature)
                plt.ylabel("Sale_Amount")

            elif plot_type == "hist":
                sns.histplot(train_df[feature], bins=40, kde=True)

            elif plot_type == "box":
                sns.boxplot(x=train_df[feature])

            elif plot_type == "kde":
                sns.kdeplot(train_df[feature], fill=True)

            plot_image = fig_to_base64()
            plt.close()
        except Exception as e:
            error_message = str(e)

    if request.method == "POST" and request.FILES:
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES["file"]
                df = pd.read_csv(uploaded_file)
                original = df.copy()

                df_clean = preprocess(df)
                df_clean = df_clean.drop(columns=["Sale_Amount"], errors="ignore")

                pipeline = load_pipeline()
                preds = pipeline.predict(df_clean)


                formatted_preds = [f"${p:.2f}" for p in preds]
                original["Predicted_Sale_Amount"] = formatted_preds

                buffer = io.StringIO()
                original.to_csv(buffer, index=False)
                buffer.seek(0)

                response = HttpResponse(buffer.getvalue(), content_type="text/csv")
                response["Content-Disposition"] = 'attachment; filename="predictions.csv"'
                return response
            except Exception as e:
                prediction_error = str(e)
        else:
            prediction_error = "Invalid file."

    required_images = {}

    try:
        model_path = Path(settings.BASE_DIR) / "ml" / "pipeline.pkl"
        pipeline = joblib.load(model_path)

        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        ohe = preprocessor.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(["Campaign_Name", "Location", "Device", "Keyword"])

        num_features = [
            "Clicks", "Impressions", "Cost", "Leads", "Conversions",
            "Conversion Rate", "Ad_Year", "Ad_Month", "Ad_DayOfWeek"
        ]

        feature_names = list(cat_features) + num_features
        importances = model.feature_importances_

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[:20], y=feature_names[:20])
        required_images["importance"] = fig_to_base64()
        plt.close()
    except:
        required_images["importance"] = None

    try:
        plt.figure(figsize=(10, 7))
        numeric_df = train_df.select_dtypes(include=["float64", "int64"])
        sns.heatmap(numeric_df.corr(), cmap="coolwarm")
        required_images["heatmap"] = fig_to_base64()
        plt.close()
    except:
        required_images["heatmap"] = None

    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df["Sale_Amount"], bins=40, kde=True)
        required_images["sale_dist"] = fig_to_base64()
        plt.close()
    except:
        required_images["sale_dist"] = None

    return render(
        request,
        "prediction/upload.html",
        {
            "form": form,
            "numeric_cols": numeric_cols,
            "plot_image": plot_image,
            "error_message": error_message,
            "prediction_error": prediction_error,
            "required_images": required_images,
        }
    )
