import pandas as pd

REQUIRED_COLUMNS = [
    'Campaign_Name', 'Location', 'Device', 'Keyword', 'Cost', 'Ad_Date', 'Clicks', 'Impressions',
    'Leads', 'Conversions'
]


def preprocess(df):
    try:
        df = df.copy()

      
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

     
        text_cols = ['Campaign_Name', 'Location', 'Device', 'Keyword']
        for col in text_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()

      
        for col in ['Cost']:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

     
        df['Ad_Date'] = df['Ad_Date'].astype(str).str.replace('/', '-', regex=False)

        parsed_dates = pd.Series(pd.NaT, index=df.index)


        parsed_ymd = pd.to_datetime(df['Ad_Date'], format="%Y-%m-%d", errors='coerce')
        parsed_dates.update(parsed_ymd.dropna())


        remaining_mask = parsed_dates.isna()
        if remaining_mask.any():
            parsed_dmy = pd.to_datetime(
                df.loc[remaining_mask, 'Ad_Date'],
                format="%d-%m-%Y",
                errors='coerce'
            )
            parsed_dates.update(parsed_dmy.dropna())


        remaining_mask = parsed_dates.isna()
        if remaining_mask.any():
            parsed_auto = pd.to_datetime(
                df.loc[remaining_mask, 'Ad_Date'],
                errors='coerce'
            )
            parsed_dates.update(parsed_auto.dropna())

        df['Ad_Date'] = parsed_dates


        if df['Ad_Date'].isna().any():
            print("⚠ Warning: Some date values could not be parsed and are set to NaT.")

        df['Ad_Year'] = df['Ad_Date'].dt.year
        df['Ad_Month'] = df['Ad_Date'].dt.month
        df['Ad_DayOfWeek'] = df['Ad_Date'].dt.dayofweek

    
        numeric_cols = ["Clicks", "Impressions", "Cost", "Leads", "Conversions"]
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


        df['Conversion Rate'] = 0
        valid = (
            df['Clicks'].notna() &
            (df['Clicks'] > 0) &
            df['Conversions'].notna()
        )
        df.loc[valid, 'Conversion Rate'] = (
            df.loc[valid, 'Conversions'] / df.loc[valid, 'Clicks']
        )

 
        df = df.drop(columns=['Ad_Date'], errors='ignore')
        df = df.drop_duplicates()

        return df

    except Exception as e:
        print("❌ ERROR in preprocessing:", str(e))
        raise e
