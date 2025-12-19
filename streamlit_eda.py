import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def requirement_1_filter(df):
    st.header("1. Interactive Dataset Filtering")

    st.subheader("First 10 Rows from Dataset")
    st.dataframe(df.head(10))

    filtered_df = df.copy()
    initial_rows = len(df)
    
    st.subheader("Apply Filters")
    
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("##### Filters for Numeric Columns")
    numeric_filters = {}
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        
        with st.expander(f"Filter: {col}"):
            range_val = st.slider(
                f"Select Range for **{col}**",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val)),
                key=f'slider_{col}'
            )
            numeric_filters[col] = range_val
            filtered_df = filtered_df[(filtered_df[col] >= range_val[0]) & (filtered_df[col] <= range_val[1])]

    st.markdown("##### Filters for Categorical Columns")
    categorical_filters = {}
    for col in categorical_columns:
        unique_vals = df[col].unique().tolist()
        
        with st.expander(f"Filter: {col}"):
            selected_vals = st.multiselect(
                f"Select Values for **{col}**",
                options=unique_vals,
                default=unique_vals,
                key=f'multiselect_{col}'
            )
            categorical_filters[col] = selected_vals
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
    st.subheader("Filtering Results")
    
    final_rows = len(filtered_df)
    st.metric(
        label="Number of Rows",
        value=f"{final_rows} (after filtering)",
        delta=f"Initial: {initial_rows}",
        delta_color="off"
    )

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)

def requirement_2_describe(df):
    st.header("2. Dataset Description and Missing Values")

    rows, cols = df.shape
    st.subheader("Dataset Dimensions")
    st.markdown(f"* Number of Rows: **{rows}**")
    st.markdown(f"* Number of Columns: **{cols}**")

    st.subheader("Data Types per Column")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column'}))

    st.subheader("Analysis of Missing Values")
    
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if not missing_data.empty:
        total_rows = len(df)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Number of Missing Values': missing_data.values,
            'Percent Missing Values': (missing_data.values / total_rows) * 100
        })
        missing_df = missing_df.sort_values(by='Percent Missing Values', ascending=False)

        st.dataframe(missing_df.set_index('Column'))
        
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Percent Missing Values',
            title='Percentage of Missing Values per Column',
            color='Percent Missing Values',
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values in this dataset! 🎉")

    st.subheader("Descriptive Statistics for Numeric Columns")
    st.dataframe(df.describe().T.rename(columns={'50%': 'median'}))


def requirement_3_univariate_numeric(df):
    st.header("3. Univariate Analysis for Numeric Columns")
    
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns in dataset.")
        return

    selected_col = st.selectbox("Select Numeric Column for Analysis:", numeric_columns)

    if selected_col:
        mean_val = df[selected_col].mean()
        median_val = df[selected_col].median()
        std_val = df[selected_col].std()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{mean_val:.2f}")
        col2.metric("Median", f"{median_val:.2f}")
        col3.metric("Standard Deviation", f"{std_val:.2f}")

        st.subheader(f"Visualization for column: **{selected_col}**")

        st.markdown("##### Histogram")
        n_bins = st.slider("Number of Bins for Histogram", min_value=10, max_value=100, value=30, step=10)
        
        fig_hist = px.histogram(
            df,
            x=selected_col,
            nbins=n_bins,
            title=f"Distribution of {selected_col}",
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("##### Box Plot")
        fig_box = px.box(
            df,
            y=selected_col,
            title=f"Box Plot for {selected_col}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

def requirement_4_univariate_categorical(df):
    st.header("4. Univariate Analysis for Categorical Columns")

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns:
        st.warning("No categorical columns (object, category) in dataset.")
        return

    selected_col = st.selectbox("Select Categorical Column for Analysis:", categorical_columns)

    if selected_col:
        st.subheader(f"Analysis for column: **{selected_col}**")
        
        frequencies = df[selected_col].value_counts().reset_index()
        frequencies.columns = ['Value', 'Absolute Frequency']
        frequencies['Percentage (%)'] = (frequencies['Absolute Frequency'] / len(df) * 100).round(2)
        
        st.markdown("##### Frequency Table")
        st.dataframe(frequencies)

        st.markdown("##### Count Plot (Bar Chart)")
        fig_count = px.bar(
            frequencies,
            x='Value',
            y='Absolute Frequency',
            title=f"Frequency of Values in {selected_col}",
            color='Absolute Frequency',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_count, use_container_width=True)


def requirement_5_advanced_analysis(df):
    st.header("5. Correlation Analysis and Outlier Detection")
    
    df_numeric = df.select_dtypes(include=np.number)
    numeric_columns = df_numeric.columns.tolist()

    if not numeric_columns or len(numeric_columns) < 2:
        st.warning("At least two numeric columns are required.")
        return

    st.subheader("Correlation Analysis between Numeric Columns")
    
    corr_matrix = df_numeric.corr()

    st.markdown("##### Correlation Matrix Heatmap")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix (Heatmap)",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("##### Bivariate Scatter Plot")
    col1, col2 = st.columns(2)
    var_x = col1.selectbox("Select Variable X:", numeric_columns, index=0)
    var_y = col2.selectbox("Select Variable Y:", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)

    if var_x and var_y:
        pearson_corr = df[[var_x, var_y]].corr().iloc[0, 1]
        st.metric(f"Pearson Correlation Coefficient ({var_x} vs {var_y})", f"{pearson_corr:.3f}")
        
        fig_scatter = px.scatter(
            df,
            x=var_x,
            y=var_y,
            title=f"Scatter Plot: {var_x} vs {var_y}",
            trendline="ols"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Outlier Detection (IQR Method)")
    outlier_results = []

    for col in numeric_columns:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_numeric[(df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)]
        num_outliers = len(outliers)
        percent_outliers = (num_outliers / len(df)) * 100
        
        outlier_results.append({
            'Column': col,
            'Number of Outliers': num_outliers,
            'Percent Outliers (%)': f"{percent_outliers:.2f}"
        })

    st.dataframe(pd.DataFrame(outlier_results).set_index('Column'))

    st.markdown("##### Outlier Visualization (Box Plot)")
    col_outlier_viz = st.selectbox("Select Column for Outlier Visualization (Box Plot):", numeric_columns)

    if col_outlier_viz:
        fig_outlier = px.box(
            df,
            y=col_outlier_viz,
            title=f"Box Plot: Outliers for {col_outlier_viz}"
        )
        st.plotly_chart(fig_outlier, use_container_width=True)



def main():
    st.set_page_config(layout="wide")
    st.title("EDA Project with Streamlit")

    df = None

    st.sidebar.header("Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.sidebar.success("File read successfully!")

            st.sidebar.header("Navigation")
            requirement_selection = st.sidebar.radio(
                "Select Requirement:",
                [f"Requirement {i}" for i in range(1, 6)]
            )

            if df is not None:
                if requirement_selection == "Requirement 1":
                    requirement_1_filter(df.copy()) 
                elif requirement_selection == "Requirement 2":
                    requirement_2_describe(df)
                elif requirement_selection == "Requirement 3":
                    requirement_3_univariate_numeric(df)
                elif requirement_selection == "Requirement 4":
                    requirement_4_univariate_categorical(df)
                elif requirement_selection == "Requirement 5":
                    requirement_5_advanced_analysis(df)

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.info("Wait for a file upload to start the analysis.")

if __name__ == '__main__':
    main()


