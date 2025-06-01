import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import streamlit as st
from itertools import combinations
import warnings
import numpy as np
from typing import Optional, List, Tuple
import os
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
from typing import Dict, List, Any, Optional, Tuple
import itertools
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import re
import umap.umap_ as umap
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import uuid
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.outliers_influence import variance_inflation_factor
import markdown2
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from typing import Optional, Dict, List, Union
from datetime import datetime

warnings.filterwarnings('ignore')
def load_and_validate_csv(file):
    """
    Load and validate a CSV file.
    
    Parameters:
    -----------
    file : UploadedFile (Streamlit) or file-like object
        The uploaded CSV file.
    
    Returns:
    --------
    tuple : (pd.DataFrame or None, str or None)
        DataFrame if successful, error message if failed.
    """
    if file is None:
        return None, "No file uploaded."
    
    try:
        # Read file with multiple encoding and delimiter attempts
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        delimiters = [',', ';', '\t']
        df = None
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding, sep=delimiter)
                    if not df.empty and len(df.columns) > 0:
                        return df, None
                except Exception:
                    continue
        
        if df is None or df.empty:
            return None, "File is empty or no valid columns found."
        if len(df.columns) == 0:
            return None, "No columns to parse from file."
        
        # Basic validation
        if df.shape[1] == 0:
            return None, "No columns detected in the file."
        if df.shape[0] == 0:
            return None, "No data rows detected in the file."
        
        return df, None
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def generate_data_summary(df: pd.DataFrame):
    try:
        summary_data = {
            'Column': [],
            'Data Type': [],
            '% Missing': [],
            'Unique Values': [],
            'Sample Values': []
        }

        for col in df.columns:
            try:
                dtype = df[col].dtype
                missing_pct = round(df[col].isnull().mean() * 100, 2)
                unique_vals = df[col].nunique()
                samples = df[col].dropna().unique()[:3]
                sample_vals = ", ".join([str(val) for val in samples])
            except Exception as col_err:
                dtype = 'Unknown'
                missing_pct = 'Error'
                unique_vals = 'Error'
                sample_vals = f"Error: {col_err}"

            summary_data['Column'].append(col)
            summary_data['Data Type'].append(dtype)
            summary_data['% Missing'].append(missing_pct)
            summary_data['Unique Values'].append(unique_vals)
            summary_data['Sample Values'].append(sample_vals)

        summary_df = pd.DataFrame(summary_data)
        return summary_df, None

    except Exception as e:
        return None, f"Error generating summary: {str(e)}"
    
def get_dataset_overview(df):
    try:
        shape = df.shape
        column_names = df.columns.tolist()
        data_types = df.dtypes.to_dict()
        missing_counts = df.isnull().sum().to_dict()
        unique_counts = df.nunique().to_dict()

        # % Missing
        missing_percent = {col: round((missing_counts[col] / len(df)) * 100, 2) for col in df.columns}

        # Sample values (for first 3 rows)
        sample_values = {col: df[col].dropna().astype(str).unique()[:3].tolist() for col in df.columns}

        # Combine into column-level overview
        column_summary = []
        for col in df.columns:
            column_summary.append({
                "Column Name": col,
                "Data Type": str(data_types[col]),
                "Missing Count": missing_counts[col],
                "Missing %": missing_percent[col],
                "Unique Values": unique_counts[col],
                "Sample Values": sample_values[col]
            })

        duplicate_rows = df.duplicated().sum()

        return {
            "Shape (rows, columns)": shape,
            "Duplicate Rows": duplicate_rows,
            "Column Summary": column_summary
        }, None

    except Exception as e:
        return None, f"Error in overview: {str(e)}"

def plot_univariate(df, column):
    import io
    import base64

    plot_type = None
    fig = None

    if pd.api.types.is_numeric_dtype(df[column]):
        plot_type = "numeric"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[column].dropna(), kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("Histogram")

        sns.boxplot(x=df[column], ax=axes[1], color='lightgreen')
        axes[1].set_title("Box Plot")

        plt.tight_layout()

    elif pd.api.types.is_object_dtype(df[column]) or df[column].nunique() < 20:
        plot_type = "categorical"
        counts = df[column].value_counts()

        if counts.shape[0] > 25:
            return {
                "too_many_categories": True,
                "message": "Too many categories to display cleanly.",
            }

        if counts.shape[0] <= 10:
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                title=f"Pie Chart of {column}",
                hole=0.4
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(y=counts.index[:15], x=counts.values[:15], ax=ax, palette="coolwarm")
            ax.set_title("Top Categories - Horizontal Bar Plot")
            ax.set_xlabel("Count")
            ax.set_ylabel(column)
            plt.tight_layout()

    else:
        return {
            "unsupported": True,
            "message": "Unsupported column type for plotting."
        }

    return {
        "fig": fig,
        "type": plot_type
    }

def suggest_target_column(df):
    suggestions = []

    # Common keywords
    common_keywords = ['target', 'label', 'class', 'y', 'outcome']

    for col in df.columns:
        col_lower = col.lower()
        if any(key in col_lower for key in common_keywords):
            suggestions.append(col)

    # If no keywords found, suggest last column if it's not fully unique
    if not suggestions:
        last_col = df.columns[-1]
        if df[last_col].nunique() < df.shape[0] * 0.3:
            suggestions.append(last_col)

    return suggestions

def render_matplotlib_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    plt.close(fig)

# 1. Categorical vs Target (Countplot)
def plot_cat_vs_target(df, feature, target):
    try:
        if df[target].nunique() > 20:
            st.warning("Target variable has too many unique values. Countplot skipped.")
            return

        top_categories = df[feature].value_counts().nlargest(15).index
        filtered_df = df[df[feature].isin(top_categories)]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x=feature, hue=target, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f"Countplot of {feature} by {target}")
        render_matplotlib_plot(fig)

    except Exception as e:
        st.error(f"Failed to plot Categorical vs Target: {e}")
    
# 2. Numeric vs Target (Boxplot, Violinplot, Swarmplot)
def plot_num_vs_target(df, feature, target, chart_type):
    try:
        if df[target].nunique() > 20:
            st.warning("Too many unique values in target for comparison.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "Boxplot":
            sns.boxplot(data=df, x=target, y=feature, ax=ax)
        elif chart_type == "Violinplot":
            sns.violinplot(data=df, x=target, y=feature, ax=ax)
        elif chart_type == "Swarmplot":
            sns.swarmplot(data=df, x=target, y=feature, ax=ax, size=3)

        ax.set_title(f"{chart_type} of {feature} by {target}")
        render_matplotlib_plot(fig)

    except Exception as e:
        st.error(f"Error plotting {chart_type}: {e}")

# 3. Class Balance (Pie and Bar)
def plot_class_balance(df, target_col):
    if target_col not in df.columns:
        return None, None

    try:
        value_counts = df[target_col].value_counts().reset_index()
        value_counts.columns = [target_col, 'Count']

        # Only show top 15 if too many categories
        if len(value_counts) > 15:
            value_counts = value_counts.iloc[:15]

        pie_fig = px.pie(value_counts, names=target_col, values='Count',
                         title=f"{target_col} Class Distribution (Pie Chart)")

        bar_fig = px.bar(value_counts, x='Count', y=target_col, orientation='h',
                         title=f"{target_col} Class Distribution (Bar Chart)")

        return pie_fig, bar_fig

    except Exception as e:
        print(f"[Error in plot_class_balance]: {e}")
        return None, None
# 4. Mean Target Value by Category
def plot_mean_target_by_category(df, feature, target):
    try:
        if df[feature].nunique() > 25:
            st.warning("Too many categories. Plotting top 15 only.")
        top_cats = df[feature].value_counts().nlargest(15).index
        filtered_df = df[df[feature].isin(top_cats)]

        grouped = filtered_df.groupby(feature)[target].mean().sort_values(ascending=False)
        fig = px.bar(x=grouped.index, y=grouped.values,
                     labels={'x': feature, 'y': f"Mean of {target}"},
                     title=f"Mean {target} per {feature}")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting mean target by category: {e}")
def explore_pairwise_relationships(
    df,
    target_col=None,
    cols=None,
    plot_type="all",
    sample_size=None,
    show_trendline=False,
    fig_size=(800, 600),
    correlation_method="pearson",
    color_scale="Viridis",
    log_x=False,
    log_y=False,
    hexbin_bins=30,
    show_outliers=False,
    return_figs=False
):
    """
    Generate bivariate plots for numeric vs. numeric relationships using Plotly.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    df = df.copy()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("DataFrame must contain at least two numeric columns.")
    
    # Validate cols
    if cols:
        if not all(col in df.columns for col in cols):
            raise ValueError("One or more specified 'cols' not found in DataFrame.")
        if not all(np.issubdtype(df[col].dtype, np.number) for col in cols):
            raise ValueError("All specified 'cols' must be numeric.")
        selected_cols = cols
    else:
        selected_cols = numeric_cols
    
    # Check for zero-variance columns
    zero_var_cols = [col for col in selected_cols if df[col].var() == 0]
    if zero_var_cols:
        raise ValueError(f"Columns {zero_var_cols} have zero variance.")
    
    # Validate target column
    if target_col:
        if target_col not in df.columns:
            raise ValueError("Target column not found in DataFrame.")
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 15:
            warnings.warn("High-cardinality numeric target; binning into quartiles.")
            df[target_col] = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
    
    # Handle missing values
    if df[selected_cols].isna().any().any():
        warnings.warn(f"Missing values found in numeric columns; dropping NA rows.")
        df = df.dropna(subset=selected_cols)
    
    # Handle sampling
    if sample_size and isinstance(sample_size, int) and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        warnings.warn(f"Sampling {sample_size} rows for performance.")
    
    # Limit pairwise combinations for large feature sets
    max_pairs = 5
    if len(selected_cols) > 5 and plot_type in ("scatter", "hexbin", "all"):
        warnings.warn(f"Too many numeric columns; selecting top {max_pairs} pairs by correlation.")
        corr = df[selected_cols].corr(method=correlation_method).abs()
        corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top_pairs = corr.unstack().sort_values(ascending=False).head(max_pairs).index
        selected_pairs = [(col1, col2) for col1, col2 in top_pairs]
    else:
        selected_pairs = list(combinations(selected_cols, 2))
    
    # Plotting helpers
    def detect_outliers(x, y):
        q1_x, q3_x = np.percentile(x, [25, 75])
        iqr_x = q3_x - q1_x
        q1_y, q3_y = np.percentile(y, [25, 75])
        iqr_y = q3_y - q1_y
        outliers = (x < q1_x - 1.5 * iqr_x) | (x > q3_x + 1.5 * iqr_x) | \
                   (y < q1_y - 1.5 * iqr_y) | (y > q3_y + 1.5 * iqr_y)
        return outliers
    
    def plot_scatter(x, y):
        fig = px.scatter(
            df, x=x, y=y, color=target_col,
            trendline="ols" if show_trendline else None,
            width=fig_size[0], height=fig_size[1],
            opacity=0.6,
            color_continuous_scale=color_scale,
            render_mode="webgl",
            hover_data=[target_col] if target_col else None
        )
        if show_outliers:
            outliers = detect_outliers(df[x], df[y])
            if outliers.any():
                fig.add_trace(go.Scattergl(
                    x=df[x][outliers], y=df[y][outliers],
                    mode='markers', marker=dict(color='red', size=10, symbol='x'),
                    name='Outliers'
                ))
        if show_trendline:
            slope, intercept, r_value, p_value, _ = stats.linregress(df[x], df[y])
            fig.add_annotation(
                text=f"R²={r_value**2:.3f}, p={p_value:.3e}",
                xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False
            )
        fig.update_layout(
            title=f"Scatter: {x} vs {y}",
            xaxis=dict(type='log' if log_x else 'linear'),
            yaxis=dict(type='log' if log_y else 'linear')
        )
        return fig
    
    def plot_correlation_heatmap():
        corr = df[selected_cols].corr(method=correlation_method)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=color_scale,
            width=fig_size[0],
            height=fig_size[1],
            title=f"{correlation_method.title()} Correlation Heatmap"
        )
        return fig
    
    def plot_joint_plot():
        fig = px.scatter_matrix(
            df, dimensions=selected_cols, color=target_col,
            width=fig_size[0], height=fig_size[1],
            opacity=0.6,
            color_continuous_scale=color_scale,
            title="Joint Plot (Scatter Matrix)"
        )
        fig.update_traces(diagonal_visible=False)
        return fig
    
    def plot_hexbin(x, y):
        fig = px.density_heatmap(
            df, x=x, y=y,
            nbinsx=hexbin_bins, nbinsy=hexbin_bins,
            color_continuous_scale=color_scale,
            width=fig_size[0], height=fig_size[1],
            title=f"Hexbin: {x} vs {y}"
        )
        fig.update_layout(
            xaxis=dict(type='log' if log_x else 'linear'),
            yaxis=dict(type='log' if log_y else 'linear')
        )
        return fig
    
    # Plot execution
    plot_list = []
    
    if plot_type in ("scatter", "all"):
        for x, y in selected_pairs:
            fig = plot_scatter(x, y)
            plot_list.append(fig)
    
    if plot_type in ("correlation", "all"):
        fig = plot_correlation_heatmap()
        plot_list.append(fig)
    
    if plot_type in ("joint", "all"):
        fig = plot_joint_plot()
        plot_list.append(fig)
    
    if plot_type in ("hexbin", "all"):
        for x, y in selected_pairs:
            fig = plot_hexbin(x, y)
            plot_list.append(fig)
    
    if return_figs:
        return plot_list
    else:
        for fig in plot_list:
            st.plotly_chart(fig, use_container_width=True)
def plot_categorical_vs_categorical(
    df,
    cat_col1,
    cat_col2,
    target_col=None,
    plot_type="all",
    normalize=None,  # None, "row", "col"
    color_scale="Cividis",
    return_figs=False,
    max_categories=15
):
    """
    Analyze and visualize relationships between two categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    cat_col1, cat_col2 : str
        Names of the two categorical columns.
    target_col : str, optional
        Target column for segmentation (categorical or low-cardinality numeric).
    plot_type : str
        Plot type: "stacked", "clustered", "mosaic", "all".
    normalize : str, optional
        Normalization for bar charts: None (counts), "row", "col".
    color_scale : str
        Plotly color scale (default: "Cividis" for colorblind-friendliness).
    return_figs : bool
        Return Plotly figures instead of displaying.
    max_categories : int
        Maximum unique categories before warning (default: 15).
    
    Returns:
    --------
    dict
        Contains figures, contingency table, and chi-square results.
    """
    # -------------------- VALIDATION --------------------
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    df = df.copy()
    
    # Validate categorical columns
    for col in [cat_col1, cat_col2]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be categorical or object type.")
    
    # Validate target column
    if target_col:
        if target_col not in df.columns:
            raise ValueError("Target column not found in DataFrame.")
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > max_categories:
            warnings.warn("High-cardinality numeric target; binning into quartiles.")
            df[target_col] = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
    
    # Handle missing values
    for col in [cat_col1, cat_col2]:
        if df[col].isna().any():
            warnings.warn(f"Missing values in '{col}'; filling with 'Unknown'.")
            df[col] = df[col].fillna("Unknown")
    
    # Check for high cardinality
    n_unique1, n_unique2 = df[cat_col1].nunique(), df[cat_col2].nunique()
    if n_unique1 > max_categories or n_unique2 > max_categories:
        warnings.warn(f"High cardinality in '{cat_col1}' ({n_unique1}) or '{cat_col2}' ({n_unique2}); plots may be cluttered.")
    
    # -------------------- CONTINGENCY TABLE --------------------
    @st.cache_data
    def compute_contingency_table(_df, col1, col2):
        table = pd.crosstab(_df[col1], _df[col2])
        if normalize == "row":
            table = table.div(table.sum(axis=1), axis=0) * 100
        elif normalize == "col":
            table = table.div(table.sum(axis=0), axis=1) * 100
        return table
    
    contingency_table = compute_contingency_table(df, cat_col1, cat_col2)
    
    # Chi-square test
    chi2_result = None
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        interpretation = "Significant dependence" if p_value < 0.05 else "No significant dependence"
        if (expected < 5).any():
            warnings.warn("Chi-square test may be unreliable due to low expected counts (<5).")
        chi2_result = {
            "statistic": chi2,
            "p_value": p_value,
            "dof": dof,
            "interpretation": interpretation
        }
    except ValueError as e:
        warnings.warn(f"Chi-square test failed: {str(e)}")
    
    # -------------------- PLOTTING HELPERS --------------------
    def plot_stacked_bar():
        data = contingency_table.reset_index().melt(id_vars=cat_col1, value_name="value", var_name=cat_col2)
        fig = px.bar(
            data,
            x=cat_col1,
            y="value",
            color=cat_col2,
            barmode="stack",
            title=f"Stacked Bar: {cat_col1} vs {cat_col2}{' (Normalized)' if normalize else ''}",
            color_discrete_sequence=px.colors.sequential.__dict__[color_scale],
            height=600
        )
        if n_unique1 > 10:
            fig.update_layout(xaxis_tickangle=45)
        if normalize:
            fig.update_yaxes(title="Percentage" if normalize else "Count")
        return fig
    
    def plot_clustered_bar():
        data = contingency_table.reset_index().melt(id_vars=cat_col1, value_name="value", var_name=cat_col2)
        fig = px.bar(
            data,
            x=cat_col1,
            y="value",
            color=cat_col2,
            barmode="group",
            title=f"Clustered Bar: {cat_col1} vs {cat_col2}{' (Normalized)' if normalize else ''}",
            color_discrete_sequence=px.colors.sequential.__dict__[color_scale],
            height=600
        )
        if n_unique1 > 10:
            fig.update_layout(xaxis_tickangle=45)
        if normalize:
            fig.update_yaxes(title="Percentage" if normalize else "Count")
        return fig
    
    def plot_mosaic():
        if n_unique1 * n_unique2 > max_categories * max_categories:
            st.warning("Skipping mosaic plot due to too many category combinations.")
            return None
        try:
            from statsmodels.graphics.mosaicplot import mosaic
            import matplotlib.pyplot as plt
            fig, _ = mosaic(df, [cat_col1, cat_col2], title=f"Mosaic Plot: {cat_col1} vs {cat_col2}")
            return fig
        except Exception as e:
            st.warning(f"Failed to generate mosaic plot: {str(e)}")
            return None
    
    # -------------------- PLOT EXECUTION --------------------
    result = {
        "figures": [],
        "contingency_table": contingency_table,
        "chi2_result": chi2_result
    }
    
    if plot_type in ("stacked", "all"):
        try:
            fig = plot_stacked_bar()
            result["figures"].append(fig)
        except Exception as e:
            st.warning(f"Failed to generate stacked bar plot: {str(e)}")
    
    if plot_type in ("clustered", "all"):
        try:
            fig = plot_clustered_bar()
            result["figures"].append(fig)
        except Exception as e:
            st.warning(f"Failed to generate clustered bar plot: {str(e)}")
    
    if plot_type in ("mosaic", "all"):
        try:
            fig = plot_mosaic()
            if fig:
                result["figures"].append(fig)
        except Exception as e:
            st.warning(f"Failed to generate mosaic plot: {str(e)}")
    
    # -------------------- RETURN RESULTS --------------------
    if return_figs:
        return result
    else:
        for fig in result["figures"]:
            if isinstance(fig, plt.Figure):
                st.pyplot(fig)
            else:
                st.plotly_chart(fig, use_container_width=True)
        st.write("**Contingency Table**")
        st.dataframe(contingency_table, use_container_width=True)
        if chi2_result:
            st.write("**Chi-Square Test Results**")
            st.write(f"Statistic: {chi2_result['statistic']:.2f}")
            st.write(f"P-value: {chi2_result['p_value']:.3e}")
            st.write(f"Degrees of Freedom: {chi2_result['dof']}")
            st.write(f"Interpretation: {chi2_result['interpretation']}")

def plot_numeric_vs_categorical(
    df,
    num_col,
    cat_col,
    target_col=None,
    plot_type="box",
    color_palette="Set2",
    return_fig=True,
    max_categories=15,
    fig_size=(10, 6),
    alpha=0.7,
    highlight_outliers=False,
    sample_size=None
):
    """
    Visualize the relationship between a numeric variable and a categorical variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    num_col : str
        Name of the numeric column (e.g., 'Age', 'Income').
    cat_col : str
        Name of the categorical column (e.g., 'Gender', 'Region').
    target_col : str, optional
        Name of the target column for hue (e.g., 'Churn').
    plot_type : str
        Plot type: 'box', 'violin', 'strip', 'swarm'.
    color_palette : str
        Seaborn color palette (e.g., 'Set2', 'Pastel1').
    return_fig : bool
        If True, return the figure for Streamlit rendering.
    max_categories : int
        Maximum unique categories allowed in cat_col (default: 15).
    fig_size : tuple
        Figure size as (width, height) in inches.
    alpha : float
        Transparency level for plot elements (0 to 1).
    highlight_outliers : bool
        If True, highlight outliers in the plot.
    sample_size : int, optional
        Number of rows to sample for performance.
    
    Returns:
    --------
    dict
        Contains the figure, status message, and warnings (if any).
    """
    # -------------------- VALIDATION --------------------
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    df = df.copy()
    
    # Validate columns
    if num_col not in df.columns:
        raise ValueError(f"Numeric column '{num_col}' not found in DataFrame.")
    if cat_col not in df.columns:
        raise ValueError(f"Categorical column '{cat_col}' not found in DataFrame.")
    
    # Validate numeric column type
    if not pd.api.types.is_numeric_dtype(df[num_col]):
        raise ValueError(f"Column '{num_col}' must be numeric.")
    
    # Validate categorical column
    if not pd.api.types.is_categorical_dtype(df[cat_col]) and not pd.api.types.is_object_dtype(df[cat_col]):
        raise ValueError(f"Column '{cat_col}' must be categorical or object type.")
    
    # Check for high cardinality
    n_unique = df[cat_col].nunique()
    if n_unique > max_categories:
        raise ValueError(f"Too many categories in '{cat_col}' ({n_unique}); max allowed is {max_categories}.")
    
    # Validate target column
    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > max_categories:
            warnings.warn("High-cardinality numeric target; binning into quartiles.")
            df[target_col] = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
    
    # Handle missing values
    cols_to_check = [num_col, cat_col]
    if target_col:
        cols_to_check.append(target_col)
    if df[cols_to_check].isna().any().any():
        warnings.warn(f"Missing values found in columns {cols_to_check}; dropping NA rows.")
        df = df.dropna(subset=cols_to_check)
    
    # Handle sampling
    if sample_size and isinstance(sample_size, int) and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        warnings.warn(f"Sampling {sample_size} rows for performance.")
    
    # Validate plot type
    valid_plot_types = ["box", "violin", "strip", "swarm"]
    if plot_type not in valid_plot_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'; must be one of {valid_plot_types}.")
    
    # -------------------- PLOTTING --------------------
    result = {"figure": None, "status": "success", "message": ""}
    
    try:
        # Set up the plot
        plt.figure(figsize=fig_size)
        sns.set_style("whitegrid")
        
        # Detect outliers if requested
        outliers = None
        if highlight_outliers:
            grouped = df.groupby(cat_col)[num_col]
            outliers = pd.Series(False, index=df.index)
            for name, group in grouped:
                q1, q3 = group.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                group_outliers = (group < lower_bound) | (group > upper_bound)
                outliers |= group_outliers.reindex(df.index, fill_value=False)
        
        # Generate the plot
        if plot_type == "box":
            sns.boxplot(x=cat_col, y=num_col, hue=target_col, data=df, palette=color_palette)
        elif plot_type == "violin":
            sns.violinplot(x=cat_col, y=num_col, hue=target_col, data=df, palette=color_palette)
        elif plot_type == "strip":
            sns.stripplot(x=cat_col, y=num_col, hue=target_col, data=df, palette=color_palette, alpha=alpha)
        elif plot_type == "swarm":
            if len(df) > 1000:
                warnings.warn("Swarm plot may be slow with large datasets; consider sampling.")
            sns.swarmplot(x=cat_col, y=num_col, hue=target_col, data=df, palette=color_palette, alpha=alpha)
        
        # Highlight outliers if requested
        if highlight_outliers and outliers.any():
            outlier_data = df[outliers]
            sns.scatterplot(
                x=cat_col, y=num_col, data=outlier_data, color="red", marker="x", s=100, label="Outliers"
            )
        
        # Adjust layout
        plt.title(f"{plot_type.capitalize()} Plot: {num_col} vs {cat_col}")
        plt.xticks(rotation=45 if n_unique > 5 else 0)
        plt.tight_layout()
        
        # Store the figure
        fig = plt.gcf()
        result["figure"] = fig
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Failed to generate plot: {str(e)}"
        warnings.warn(result["message"])
    
    # -------------------- RETURN RESULTS --------------------
    if return_fig:
        return result
    else:
        if result["figure"]:
            st.pyplot(result["figure"])
        if result["message"]:
            st.warning(result["message"])

class SmartPlottingEngine:
    """
    AI-based plotting engine that intelligently suggests and configures plots
    based on data types and user selections.
    """
    
    def __init__(self):
        self.plot_configs = self._initialize_plot_configs()
        self.data_type_mapping = {
            'numeric': ['int64', 'float64', 'int32', 'float32', 'number'],
            'categorical': ['object', 'category', 'string', 'bool'],
            'datetime': ['datetime64', 'timedelta64']
        }
    
    def _initialize_plot_configs(self) -> Dict[str, Dict]:
        """Initialize configuration for each plot type with required/optional parameters."""
        return {
            'scatter': {
                'description': 'Explore relationships between two numeric variables',
                'required_data': ['numeric_x', 'numeric_y'],
                'optional_params': ['hue', 'size', 'style', 'regression', 'log_scales', 'transparency', 'point_size'],
                'incompatible_with': [],
                'best_for': 'numeric vs numeric relationships'
            },
            'line': {
                'description': 'Show trends over time or ordered categories',
                'required_data': ['x_axis', 'numeric_y'],
                'optional_params': ['hue', 'style', 'transparency', 'markers'],
                'incompatible_with': ['size', 'point_size'],
                'best_for': 'time series or ordered data'
            },
            'histogram': {
                'description': 'Distribution of a single numeric variable',
                'required_data': ['numeric_x'],
                'optional_params': ['hue', 'bins', 'density', 'transparency'],
                'incompatible_with': ['y_axis', 'size', 'style', 'regression'],
                'best_for': 'single variable distribution'
            },
            'box': {
                'description': 'Compare distributions across categories',
                'required_data': ['categorical_x', 'numeric_y'],
                'optional_params': ['hue', 'orientation', 'outliers'],
                'incompatible_with': ['size', 'style', 'regression', 'point_size'],
                'best_for': 'numeric distribution by category'
            },
            'violin': {
                'description': 'Detailed distribution comparison across categories',
                'required_data': ['categorical_x', 'numeric_y'],
                'optional_params': ['hue', 'orientation', 'inner_plot'],
                'incompatible_with': ['size', 'style', 'regression', 'point_size'],
                'best_for': 'detailed distribution comparison'
            },
            'bar': {
                'description': 'Compare aggregated values across categories',
                'required_data': ['categorical_x', 'numeric_y'],
                'optional_params': ['hue', 'aggregation', 'orientation', 'error_bars'],
                'incompatible_with': ['size', 'style', 'regression', 'point_size'],
                'best_for': 'comparing aggregated values'
            },
            'countplot': {
                'description': 'Count occurrences of categorical values',
                'required_data': ['categorical_x'],
                'optional_params': ['hue', 'orientation'],
                'incompatible_with': ['y_axis', 'size', 'style', 'regression', 'aggregation'],
                'best_for': 'categorical value counts'
            },
            'heatmap': {
                'description': 'Show relationships in a matrix format',
                'required_data': ['categorical_x', 'categorical_y', 'numeric_value'],
                'optional_params': ['aggregation', 'color_map', 'annotations'],
                'incompatible_with': ['hue', 'size', 'style', 'regression', 'transparency', 'point_size'],
                'best_for': 'matrix relationships'
            },
            'kde': {
                'description': 'Smooth density estimation',
                'required_data': ['numeric_x'],
                'optional_params': ['hue', 'bivariate_y', 'transparency'],
                'incompatible_with': ['size', 'style', 'regression', 'point_size'],
                'best_for': 'smooth distributions'
            },
            'jointplot': {
                'description': 'Bivariate relationship with marginal distributions',
                'required_data': ['numeric_x', 'numeric_y'],
                'optional_params': ['hue', 'kind', 'regression'],
                'incompatible_with': ['size', 'style', 'transparency', 'point_size'],
                'best_for': 'detailed bivariate analysis'
            },
            'pairplot': {
                'description': 'All pairwise relationships in dataset',
                'required_data': ['multiple_numeric'],
                'optional_params': ['hue', 'diagonal_kind'],
                'incompatible_with': ['x_axis', 'y_axis', 'size', 'style', 'regression'],
                'best_for': 'comprehensive variable relationships'
            }
        }
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze DataFrame and categorize columns by data type."""
        analysis = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'all': list(df.columns)
        }
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['datetime'].append(col)
            else:
                analysis['categorical'].append(col)
        
        return analysis
    
    def suggest_plots(self, df: pd.DataFrame, x_col: str = None, y_col: str = None) -> List[Dict]:
        """AI-based plot suggestions based on selected columns."""
        data_analysis = self.analyze_data(df)
        suggestions = []
        
        if not x_col:
            # No selection yet - suggest based on data types available
            if data_analysis['numeric']:
                suggestions.extend(['histogram', 'kde'])
            if data_analysis['categorical']:
                suggestions.append('countplot')
            if len(data_analysis['numeric']) >= 2:
                suggestions.extend(['scatter', 'pairplot', 'jointplot'])
            return [{'plot': plot, **self.plot_configs[plot]} for plot in suggestions]
        
        x_type = 'numeric' if x_col in data_analysis['numeric'] else 'categorical'
        
        if not y_col:
            # Only X selected
            if x_type == 'numeric':
                suggestions = ['histogram', 'kde']
            else:
                suggestions = ['countplot']
        else:
            # Both X and Y selected
            y_type = 'numeric' if y_col in data_analysis['numeric'] else 'categorical'
            
            if x_type == 'numeric' and y_type == 'numeric':
                suggestions = ['scatter', 'line', 'kde', 'jointplot']
            elif x_type == 'categorical' and y_type == 'numeric':
                suggestions = ['box', 'violin', 'bar']
            elif x_type == 'numeric' and y_type == 'categorical':
                suggestions = ['box', 'violin', 'bar']  # Will swap axes
            elif x_type == 'categorical' and y_type == 'categorical':
                suggestions = ['heatmap', 'countplot']
        
        return [{'plot': plot, **self.plot_configs[plot]} for plot in suggestions if plot in self.plot_configs]
    
    def get_relevant_params(self, plot_type: str) -> Dict[str, List[str]]:
        """Get relevant parameters for a specific plot type."""
        if plot_type not in self.plot_configs:
            return {'required': [], 'optional': [], 'incompatible': []}
        
        config = self.plot_configs[plot_type]
        return {
            'required': config.get('required_data', []),
            'optional': config.get('optional_params', []),
            'incompatible': config.get('incompatible_with', [])
        }
    
    def create_smart_plot(self, df: pd.DataFrame, plot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create plot with smart parameter handling."""
        result = {"figure": None, "status": "success", "message": ""}
        
        try:
            plot_type = plot_config['plot_type']
            params = plot_config.get('params', {})
            
            # Apply sampling if needed
            working_df = df.copy()
            if params.get('sample_size') and len(working_df) > params['sample_size']:
                working_df = working_df.sample(n=params['sample_size'], random_state=42)
                result["message"] += f"Sampled {params['sample_size']} rows for performance. "
            
            # Apply filters
            if params.get('filter_condition'):
                working_df = self._apply_filters(working_df, params['filter_condition'])
            
            # Remove missing values for selected columns
            cols_to_check = [col for col in [params.get('x_col'), params.get('y_col'), 
                           params.get('hue_col')] if col and col in working_df.columns]
            if cols_to_check:
                initial_len = len(working_df)
                working_df = working_df.dropna(subset=cols_to_check)
                if len(working_df) < initial_len:
                    result["message"] += f"Removed {initial_len - len(working_df)} rows with missing values. "
            
            # Create the plot
            fig = self._create_plot_by_type(working_df, plot_type, params)
            result["figure"] = fig
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error creating plot: {str(e)}"
        
        return result
    
    def _apply_filters(self, df: pd.DataFrame, filter_condition: Dict) -> pd.DataFrame:
        """Apply filtering conditions to DataFrame."""
        if 'column' in filter_condition and filter_condition['column'] in df.columns:
            col = filter_condition['column']
            if 'value' in filter_condition:
                return df[df[col] == filter_condition['value']]
            elif 'range' in filter_condition and len(filter_condition['range']) == 2:
                min_val, max_val = filter_condition['range']
                return df[(df[col] >= min_val) & (df[col] <= max_val)]
        return df
    
    def _create_plot_by_type(self, df: pd.DataFrame, plot_type: str, params: Dict) -> plt.Figure:
        """Create specific plot type with given parameters."""
        plt.figure(figsize=params.get('figsize', (10, 6)))
        sns.set_style("whitegrid")
        
        x_col = params.get('x_col')
        y_col = params.get('y_col')
        hue_col = params.get('hue_col')
        
        if plot_type == 'scatter':
            sns.scatterplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                size=params.get('size_col'),
                style=params.get('style_col'),
                alpha=params.get('alpha', 0.7),
                s=params.get('point_size', 50)
            )
            if params.get('show_regression', False):
                sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="red", ci=None)
        
        elif plot_type == 'line':
            sns.lineplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                style=params.get('style_col'),
                alpha=params.get('alpha', 0.8),
                markers=params.get('markers', True)
            )
        
        elif plot_type == 'histogram':
            sns.histplot(
                data=df, x=x_col, hue=hue_col,
                bins=params.get('bins', 'auto'),
                stat=params.get('stat', 'count'),
                alpha=params.get('alpha', 0.7)
            )
        
        elif plot_type == 'box':
            sns.boxplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                orient=params.get('orientation', 'v'),
                showfliers=params.get('show_outliers', True)
            )
        
        elif plot_type == 'violin':
            sns.violinplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                orient=params.get('orientation', 'v'),
                inner=params.get('inner', 'box')
            )
        
        elif plot_type == 'bar':
            if params.get('aggregation', 'mean') != 'count':
                agg_df = df.groupby(x_col)[y_col].agg(params.get('aggregation', 'mean')).reset_index()
                sns.barplot(data=agg_df, x=x_col, y=y_col, hue=hue_col)
            else:
                sns.countplot(data=df, x=x_col, hue=hue_col)
        
        elif plot_type == 'countplot':
            sns.countplot(data=df, x=x_col, hue=hue_col, orient=params.get('orientation', 'v'))
        
        elif plot_type == 'heatmap':
            if params.get('value_col'):
                pivot_df = df.pivot_table(
                    values=params['value_col'], 
                    index=x_col, 
                    columns=y_col, 
                    aggfunc=params.get('aggregation', 'mean')
                )
                sns.heatmap(
                    pivot_df, 
                    annot=params.get('show_annotations', True),
                    cmap=params.get('colormap', 'viridis'),
                    fmt='.2f'
                )
        
        elif plot_type == 'kde':
            if y_col and params.get('bivariate', False):
                sns.kdeplot(data=df, x=x_col, y=y_col, hue=hue_col)
            else:
                sns.kdeplot(data=df, x=x_col, hue=hue_col, alpha=params.get('alpha', 0.7))
        
        elif plot_type == 'jointplot':
            g = sns.jointplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                kind=params.get('kind', 'scatter')
            )
            return g.figure
        
        elif plot_type == 'pairplot':
            numeric_cols = params.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
            plot_df = df[numeric_cols + ([hue_col] if hue_col else [])]
            g = sns.pairplot(plot_df, hue=hue_col, diag_kind=params.get('diag_kind', 'hist'))
            return g.figure
        
        # Apply log scales if requested
        if params.get('log_x', False):
            plt.xscale('log')
        if params.get('log_y', False):
            plt.yscale('log')
        
        # Set title and labels
        plt.title(f"{plot_type.title()}: {x_col}" + (f" vs {y_col}" if y_col else ""))
        plt.tight_layout()
        
        return plt.gcf()


def create_smart_plotting_interface(df: pd.DataFrame, target: str = None):
    """Create the smart plotting interface for Streamlit."""
    if df is None or df.empty:
        st.error("No data available for plotting.")
        return
    
    # Initialize the smart plotting engine
    engine = SmartPlottingEngine()
    data_analysis = engine.analyze_data(df)
    
    st.subheader("AI-Powered Smart Plotting")
    st.markdown("*Intelligent plot suggestions based on your data selection*")
    
    # Step 1: Column Selection
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox(
            "Select X-axis Column",
            options=[''] + data_analysis['all'],
            help="Choose your primary variable"
        )
    
    with col2:
        y_options = [''] + [col for col in data_analysis['all'] if col != x_col]
        y_col = st.selectbox(
            "Select Y-axis Column (Optional)",
            options=y_options,
            help="Choose secondary variable for relationships"
        )
    
    # Step 2: AI Plot Suggestions
    if x_col:
        suggestions = engine.suggest_plots(df, x_col, y_col if y_col else None)
        
        if suggestions:
            st.markdown("### Recommended Plots")
            
            # Create tabs for each suggestion
            plot_names = [s['plot'] for s in suggestions]
            plot_tabs = st.tabs([f"{plot.title()}" for plot in plot_names])
            
            for idx, (tab, suggestion) in enumerate(zip(plot_tabs, suggestions)):
                with tab:
                    plot_type = suggestion['plot']
                    
                    # Show description
                    st.info(f"**{suggestion['description']}** - {suggestion['best_for']}")
                    
                    # Get relevant parameters for this plot
                    relevant_params = engine.get_relevant_params(plot_type)
                    
                    # Create form for this specific plot
                    with st.form(key=f"plot_form_{plot_type}"):
                        params = {'plot_type': plot_type, 'x_col': x_col, 'y_col': y_col}
                        
                        # Conditional parameter inputs based on plot type
                        if 'hue' in relevant_params['optional']:
                            hue_options = [''] + [col for col in data_analysis['categorical'] 
                                                if col not in [x_col, y_col] and col != target]
                            if hue_options:
                                params['hue_col'] = st.selectbox(
                                    "Color by (Hue)", 
                                    options=hue_options,
                                    key=f"hue_{plot_type}"
                                )
                        
                        if 'size' in relevant_params['optional']:
                            size_options = [''] + [col for col in data_analysis['numeric'] 
                                                 if col not in [x_col, y_col]]
                            if size_options:
                                params['size_col'] = st.selectbox(
                                    "Size by", 
                                    options=size_options,
                                    key=f"size_{plot_type}"
                                )
                        
                        if 'style' in relevant_params['optional']:
                            style_options = [''] + [col for col in data_analysis['categorical'] 
                                                  if col not in [x_col, y_col, params.get('hue_col')] and col != target]
                            if style_options:
                                params['style_col'] = st.selectbox(
                                    "Style by", 
                                    options=style_options,
                                    key=f"style_{plot_type}"
                                )
                        
                        # Plot-specific parameters
                        if plot_type == 'scatter':
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                params['alpha'] = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"alpha_{plot_type}")
                            with col2:
                                params['point_size'] = st.slider("Point Size", 10, 200, 50, key=f"size_{plot_type}")
                            with col3:
                                params['show_regression'] = st.checkbox("Regression Line", key=f"reg_{plot_type}")
                        
                        elif plot_type == 'histogram':
                            col1, col2 = st.columns(2)
                            with col1:
                                params['bins'] = st.slider("Number of Bins", 5, 100, 30, key=f"bins_{plot_type}")
                            with col2:
                                params['stat'] = st.selectbox("Statistic", ['count', 'density', 'probability'], key=f"stat_{plot_type}")
                        
                        elif plot_type in ['box', 'violin']:
                            col1, col2 = st.columns(2)
                            with col1:
                                params['orientation'] = st.selectbox("Orientation", ['vertical', 'horizontal'], key=f"orient_{plot_type}")
                            if plot_type == 'box':
                                with col2:
                                    params['show_outliers'] = st.checkbox("Show Outliers", True, key=f"outliers_{plot_type}")
                            else:
                                with col2:
                                    params['inner'] = st.selectbox("Inner Plot", ['box', 'quart', 'point', 'stick'], key=f"inner_{plot_type}")
                        
                        elif plot_type == 'bar':
                            params['aggregation'] = st.selectbox(
                                "Aggregation Method", 
                                ['mean', 'sum', 'count', 'median'],
                                key=f"agg_{plot_type}"
                            )
                        
                        elif plot_type == 'heatmap':
                            value_options = [col for col in data_analysis['numeric'] if col not in [x_col, y_col]]
                            if value_options:
                                params['value_col'] = st.selectbox("Value Column", value_options, key=f"val_{plot_type}")
                                params['aggregation'] = st.selectbox("Aggregation", ['mean', 'sum', 'count'], key=f"agg_{plot_type}")
                                params['show_annotations'] = st.checkbox("Show Values", True, key=f"annot_{plot_type}")
                        
                        elif plot_type == 'kde':
                            if y_col:
                                params['bivariate'] = st.checkbox("Bivariate Plot", key=f"bivar_{plot_type}")
                        
                        elif plot_type == 'jointplot':
                            params['kind'] = st.selectbox(
                                "Plot Kind", 
                                ['scatter', 'reg', 'resid', 'kde', 'hex'],
                                key=f"kind_{plot_type}"
                            )
                        
                        elif plot_type == 'pairplot':
                            available_numeric = [col for col in data_analysis['numeric'] if col != target]
                            if len(available_numeric) > 2:
                                params['columns'] = st.multiselect(
                                    "Select Columns", 
                                    available_numeric,
                                    default=available_numeric[:4],
                                    key=f"cols_{plot_type}"
                                )
                        
                        # Common parameters
                        with st.expander("Advanced Options"):
                            col1, col2 = st.columns(2)
                            with col1:
                                if len(df) > 1000:
                                    params['sample_size'] = st.slider(
                                        "Sample Size", 
                                        100, 
                                        min(5000, len(df)), 
                                        min(1000, len(df)),
                                        key=f"sample_{plot_type}"
                                    )
                                
                                if x_col in data_analysis['numeric']:
                                    params['log_x'] = st.checkbox("Log X-axis", key=f"logx_{plot_type}")
                            
                            with col2:
                                if y_col and y_col in data_analysis['numeric']:
                                    params['log_y'] = st.checkbox("Log Y-axis", key=f"logy_{plot_type}")
                        
                        # Submit button
                        submit = st.form_submit_button(f"Generate {plot_type.title()} Plot")
                    
                    # Generate plot
                    if submit:
                        with st.spinner(f"Creating {plot_type} plot..."):
                            result = engine.create_smart_plot(df, {'plot_type': plot_type, 'params': params})
                            
                            if result['status'] == 'success' and result['figure']:
                                st.pyplot(result['figure'])
                                plt.close()  # Clean up
                                
                                if result['message']:
                                    st.info(result['message'])
                            else:
                                st.error(result['message'])
        else:
            st.warning("Please select at least one column to see plot suggestions.")
    else:
        st.info("Start by selecting a column to see AI-powered plot recommendations!")
        
        # Show data overview
        st.markdown("### Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Numeric Columns", len(data_analysis['numeric']))
        with col3:
            st.metric("Categorical Columns", len(data_analysis['categorical']))
        
        if data_analysis['numeric']:
            st.markdown("**Numeric Columns:**")
            st.write(", ".join(data_analysis['numeric']))
        
        if data_analysis['categorical']:
            st.markdown("**Categorical Columns:**")
            st.write(", ".join(data_analysis['categorical']))

class AdvancedAnalysisEngine:
    """
    AI-powered advanced analysis engine for discovering hidden patterns,
    structural relationships, and complex interactions in data.
    """
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.analysis_cache = {}
        self.recommendations = {}
    
    def _safe_numeric_conversion(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Safely convert series to numeric, return converted series and success flag."""
        try:
            converted = pd.to_numeric(series, errors='coerce')
            # Check if conversion was meaningful (not all NaN)
            if converted.notna().sum() > 0 and converted.var() > 1e-10:
                return converted, True
            return series, False
        except:
            return series, False
    
    def _get_robust_numeric_columns(self, df: pd.DataFrame, min_non_null_ratio: float = 0.1) -> List[str]:
        """Get truly numeric columns with sufficient non-null values."""
        numeric_cols = []
        
        for col in df.columns:
            try:
                # First check if it's already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for sufficient non-null values and variance
                    non_null_data = df[col].dropna()
                    if (len(non_null_data) >= max(10, len(df) * min_non_null_ratio) and 
                        non_null_data.var() > 1e-10):
                        numeric_cols.append(col)
                else:
                    # Try to convert to numeric
                    converted, success = self._safe_numeric_conversion(df[col])
                    if success:
                        non_null_data = converted.dropna()
                        if (len(non_null_data) >= max(10, len(df) * min_non_null_ratio) and 
                            non_null_data.var() > 1e-10):
                            numeric_cols.append(col)
            except:
                continue
                
        return numeric_cols
    
    def _get_robust_categorical_columns(self, df: pd.DataFrame, max_unique_ratio: float = 0.5) -> List[str]:
        """Get categorical columns with reasonable cardinality."""
        categorical_cols = []
        
        for col in df.columns:
            try:
                # Skip if already identified as numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                # Check unique values ratio
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    unique_ratio = non_null_data.nunique() / len(non_null_data)
                    if unique_ratio <= max_unique_ratio and non_null_data.nunique() >= 2:
                        categorical_cols.append(col)
            except:
                continue
                
        return categorical_cols
    
    def analyze_data_structure(self, df: pd.DataFrame, target: str = None) -> Dict[str, Any]:
        """Comprehensive data structure analysis with AI recommendations."""
        analysis = {
            'basic_info': {},
            'missing_patterns': {},
            'data_types': {},
            'ai_recommendations': {},
            'complexity_score': 0
        }
        
        try:
            # Basic information
            analysis['basic_info'] = {
                'n_rows': len(df),
                'n_features': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Robust data type categorization
            numeric_cols = self._get_robust_numeric_columns(df)
            categorical_cols = self._get_robust_categorical_columns(df)
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # High cardinality check with safety
            high_cardinality = []
            for col in categorical_cols:
                try:
                    if df[col].nunique() > min(50, len(df) * 0.05):
                        high_cardinality.append(col)
                except:
                    continue
            
            # Low variance check with safety
            low_variance = []
            for col in numeric_cols:
                try:
                    if df[col].var() < 1e-6:
                        low_variance.append(col)
                except:
                    continue
            
            analysis['data_types'] = {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols,
                'high_cardinality': high_cardinality,
                'low_variance': low_variance
            }
            
            # Missing value patterns with error handling
            try:
                missing_info = df.isnull().sum()
                analysis['missing_patterns'] = {
                    'missing_counts': missing_info.to_dict(),
                    'missing_percentages': (missing_info / len(df) * 100).to_dict(),
                    'completely_missing': missing_info[missing_info == len(df)].index.tolist(),
                    'mostly_missing': missing_info[missing_info > len(df) * 0.8].index.tolist(),
                    'pattern_correlation': self._analyze_missing_patterns(df)
                }
            except Exception as e:
                analysis['missing_patterns'] = {
                    'missing_counts': {},
                    'missing_percentages': {},
                    'completely_missing': [],
                    'mostly_missing': [],
                    'pattern_correlation': {'error': str(e)}
                }
            
            # AI Recommendations
            analysis['ai_recommendations'] = self._generate_ai_recommendations(analysis, target)
            
            # Complexity score (0-100) with safety checks
            complexity_factors = []
            try:
                complexity_factors.append(min(analysis['basic_info']['n_features'] / 100, 1) * 30)
                complexity_factors.append(min(len(analysis['data_types']['high_cardinality']) / 10, 1) * 20)
                missing_ratio = sum(1 for x in analysis['missing_patterns']['missing_percentages'].values() if x > 0) / max(analysis['basic_info']['n_features'], 1)
                complexity_factors.append(min(missing_ratio, 1) * 25)
                type_ratio = len(analysis['data_types']['categorical']) / max(analysis['basic_info']['n_features'], 1)
                complexity_factors.append(min(type_ratio, 1) * 25)
                analysis['complexity_score'] = sum(complexity_factors)
            except:
                analysis['complexity_score'] = 50  # Default moderate complexity
            
        except Exception as e:
            analysis['error'] = f"Error in data structure analysis: {str(e)}"
            
        return analysis
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data with robust error handling."""
        try:
            missing_df = df.isnull()
            
            if missing_df.empty or missing_df.sum().sum() == 0:
                return {
                    'correlated_missingness': [],
                    'missing_combinations': []
                }
            
            # Find columns with correlated missingness
            high_corr_pairs = []
            try:
                missing_corr = missing_df.corr()
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_val = missing_corr.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                            high_corr_pairs.append({
                                'col1': missing_corr.columns[i],
                                'col2': missing_corr.columns[j],
                                'correlation': corr_val
                            })
            except:
                pass
            
            return {
                'correlated_missingness': high_corr_pairs,
                'missing_combinations': self._find_missing_combinations(missing_df)
            }
        except Exception as e:
            return {
                'correlated_missingness': [],
                'missing_combinations': [],
                'error': str(e)
            }
    
    def _find_missing_combinations(self, missing_df: pd.DataFrame) -> List[Dict]:
        """Find common combinations of missing values with error handling."""
        try:
            if missing_df.empty:
                return []
                
            # Get most common missing patterns (limited for performance)
            missing_patterns = missing_df.value_counts().head(min(10, len(missing_df)))
            combinations = []
            
            for pattern, count in missing_patterns.items():
                if isinstance(pattern, tuple) and any(pattern):  # Only patterns with some missing values
                    missing_cols = [col for col, is_missing in zip(missing_df.columns, pattern) if is_missing]
                    if missing_cols:  # Ensure we have missing columns
                        combinations.append({
                            'columns': missing_cols,
                            'count': count,
                            'percentage': count / len(missing_df) * 100
                        })
            
            return combinations
        except:
            return []
    
    def _generate_ai_recommendations(self, analysis: Dict, target: str = None) -> List[Dict]:
        """Generate AI-powered recommendations based on data analysis."""
        recommendations = []
        
        try:
            # Missing data recommendations
            missing_pct = analysis.get('missing_patterns', {}).get('missing_percentages', {})
            high_missing = [col for col, pct in missing_pct.items() if pct > 50]
            
            if high_missing:
                recommendations.append({
                    'type': 'data_quality',
                    'priority': 'high',
                    'title': 'High Missing Data Detected',
                    'description': f"Columns {high_missing[:3]} have >50% missing values",
                    'action': 'Consider dropping these columns or investigating why data is missing',
                    'columns': high_missing
                })
            
            # High cardinality recommendations
            high_cardinality = analysis.get('data_types', {}).get('high_cardinality', [])
            if high_cardinality:
                recommendations.append({
                    'type': 'feature_engineering',
                    'priority': 'medium',
                    'title': 'High Cardinality Features',
                    'description': f"Features with many unique values detected",
                    'action': 'Consider grouping rare categories or using embedding techniques',
                    'columns': high_cardinality
                })
            
            # Dimensionality recommendations
            n_features = analysis.get('basic_info', {}).get('n_features', 0)
            n_rows = analysis.get('basic_info', {}).get('n_rows', 1)
            
            if n_features > n_rows * 0.1 and n_features > 10:
                recommendations.append({
                    'type': 'dimensionality',
                    'priority': 'high',
                    'title': 'High Dimensionality Dataset',
                    'description': f"Many features ({n_features}) relative to samples ({n_rows})",
                    'action': 'Consider PCA, feature selection, or regularization techniques',
                    'suggested_analysis': ['pca', 'mutual_info']
                })
            
            # Complexity-based recommendations
            complexity = analysis.get('complexity_score', 0)
            if complexity > 70:
                recommendations.append({
                    'type': 'analysis_strategy',
                    'priority': 'high',
                    'title': 'Complex Dataset Detected',
                    'description': f"Dataset complexity score: {complexity:.1f}/100",
                    'action': 'Recommend starting with dimensionality reduction and interaction analysis',
                    'suggested_analysis': ['pca', 'umap', 'mutual_info', 'interactions']
                })
        except Exception as e:
            recommendations.append({
                'type': 'error',
                'priority': 'high',
                'title': 'Analysis Error',
                'description': f"Error generating recommendations: {str(e)}",
                'action': 'Check data quality and format'
            })
        
        return recommendations
    
    def create_missingness_analysis(self, df: pd.DataFrame, advanced_options: Dict = None) -> Dict[str, Any]:
        """Create comprehensive missing data visualization and analysis."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        if advanced_options is None:
            advanced_options = {}
        
        try:
            missing_data = df.isnull()
            missing_counts = missing_data.sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            # Check if there's any missing data
            if missing_counts.sum() == 0:
                result["message"] = "No missing data found in the dataset."
                result["insights"] = {
                    'total_missing_features': 0,
                    'worst_feature': None,
                    'worst_percentage': 0,
                    'summary_stats': {
                        'features_with_missing': 0,
                        'average_missing_pct': 0,
                        'total_missing_cells': 0,
                        'completely_empty_features': 0
                    }
                }
                return result
            
            # 1. Missing data heatmap
            if advanced_options.get('show_heatmap', True):
                try:
                    cols_with_missing = missing_counts[missing_counts > 0].index
                    if len(cols_with_missing) > 0:
                        fig_heatmap, ax = plt.subplots(figsize=(12, min(8, max(4, len(cols_with_missing) * 0.3))))
                        
                        # Limit columns for visualization if too many
                        if len(cols_with_missing) > 50:
                            cols_with_missing = cols_with_missing[:50]
                            result["message"] += "Limited heatmap to top 50 features with missing data. "
                        
                        sns.heatmap(missing_data[cols_with_missing].T, 
                                  cbar=True, cmap='RdYlBu_r', 
                                  yticklabels=True, xticklabels=False, ax=ax)
                        ax.set_title('Missing Data Pattern Heatmap')
                        ax.set_ylabel('Features')
                        ax.set_xlabel('Samples')
                        plt.tight_layout()
                        result["figures"]["heatmap"] = fig_heatmap
                except Exception as e:
                    result["message"] += f"Heatmap generation failed: {str(e)[:100]}... "
            
            # 2. Missing data bar chart
            try:
                missing_features = missing_counts[missing_counts > 0]
                if len(missing_features) > 0:
                    fig_bar, ax = plt.subplots(figsize=(10, max(6, len(missing_features) * 0.2)))
                    
                    missing_data_df = pd.DataFrame({
                        'Column': missing_features.index,
                        'Missing_Count': missing_features.values,
                        'Missing_Percentage': missing_percentages[missing_features.index].values
                    })
                    
                    # Sort by percentage and limit if too many
                    missing_data_df = missing_data_df.sort_values('Missing_Percentage', ascending=True)
                    if len(missing_data_df) > 30:
                        missing_data_df = missing_data_df.tail(30)  # Show worst 30
                        result["message"] += "Limited bar chart to top 30 features with highest missing percentages. "
                    
                    # Create horizontal bar plot
                    bars = ax.barh(missing_data_df['Column'], missing_data_df['Missing_Percentage'])
                    ax.set_xlabel('Missing Percentage (%)')
                    ax.set_title('Missing Data by Feature')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add percentage labels
                    for i, (idx, row) in enumerate(missing_data_df.iterrows()):
                        ax.text(row['Missing_Percentage'] + 0.5, i, f"{row['Missing_Percentage']:.1f}%", 
                                va='center', ha='left', fontsize=8)
                    
                    plt.tight_layout()
                    result["figures"]["bar_chart"] = fig_bar
            except Exception as e:
                result["message"] += f"Bar chart generation failed: {str(e)[:100]}... "
            
            # 3. Missing data correlation (if requested)
            if advanced_options.get('show_correlation', False):
                try:
                    cols_with_missing = missing_counts[missing_counts > 0].index
                    if len(cols_with_missing) > 1:
                        missing_corr = missing_data[cols_with_missing].corr()
                        
                        if not missing_corr.isnull().all().all():
                            fig_corr, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(missing_corr, annot=True, cmap='RdBu_r', 
                                      center=0, square=True, fmt='.2f', ax=ax)
                            ax.set_title('Missing Data Correlation Matrix')
                            plt.tight_layout()
                            result["figures"]["correlation"] = fig_corr
                except Exception as e:
                    result["message"] += f"Correlation matrix generation failed: {str(e)[:100]}... "
            
            # Generate insights
            result["insights"] = {
                'total_missing_features': len(missing_counts[missing_counts > 0]),
                'worst_feature': missing_counts.idxmax() if missing_counts.max() > 0 else None,
                'worst_percentage': float(missing_percentages.max()),
                'summary_stats': {
                    'features_with_missing': int(len(missing_counts[missing_counts > 0])),
                    'average_missing_pct': float(missing_percentages[missing_percentages > 0].mean() if len(missing_percentages[missing_percentages > 0]) > 0 else 0),
                    'total_missing_cells': int(missing_data.sum().sum()),
                    'completely_empty_features': int(len(missing_counts[missing_counts == len(df)]))
                }
            }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in missingness analysis: {str(e)}"
        
        return result
    
    def create_pca_analysis(self, df: pd.DataFrame, target: str = None, 
                          scaling_method: str = 'standard', n_components: int = None,
                          advanced_options: Dict = None) -> Dict[str, Any]:
        """Perform comprehensive PCA analysis with visualizations."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        if advanced_options is None:
            advanced_options = {}
        
        try:
            # Get robust numeric columns
            numeric_cols = self._get_robust_numeric_columns(df)
            if target and target in numeric_cols:
                numeric_cols.remove(target)
            
            if len(numeric_cols) < 2:
                result["status"] = "error"
                result["message"] = f"Need at least 2 numeric features for PCA. Found {len(numeric_cols)} valid numeric columns."
                return result
            
            # Prepare data with robust handling
            X = df[numeric_cols].copy()
            
            # Convert to numeric and handle missing values
            for col in numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Remove rows with too many missing values
            X = X.dropna(thresh=len(numeric_cols) * 0.5)  # Keep rows with at least 50% non-null
            
            if len(X) < 10:
                result["status"] = "error"
                result["message"] = f"Insufficient data after cleaning: {len(X)} samples remaining"
                return result
            
            # Fill remaining missing values
            X = X.fillna(X.median())
            
            # Check for constant columns
            non_constant_cols = []
            for col in numeric_cols:
                if X[col].var() > 1e-10:
                    non_constant_cols.append(col)
            
            if len(non_constant_cols) < 2:
                result["status"] = "error"
                result["message"] = f"Need at least 2 non-constant features. Found {len(non_constant_cols)}."
                return result
            
            X = X[non_constant_cols]
            
            # Scale data
            try:
                scaler = self.scalers.get(scaling_method, StandardScaler())
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"Scaling failed: {str(e)}"
                return result
            
            # Determine number of components
            max_components = min(len(non_constant_cols), len(X))
            if n_components is None or n_components > max_components:
                n_components = max_components
            
            # Fit PCA
            try:
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"PCA fitting failed: {str(e)}"
                return result
            
            # 1. Variance explained plot (Scree plot)
            try:
                fig_variance, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Individual variance
                ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                       pca.explained_variance_ratio_)
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Proportion of Variance Explained')
                ax1.set_title('Scree Plot')
                ax1.grid(alpha=0.3)
                
                # Cumulative variance
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                ax2.plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Variance Explained')
                ax2.set_title('Cumulative Variance Explained')
                ax2.grid(alpha=0.3)
                
                # Add reference lines
                ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%')
                ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95%')
                ax2.legend()
                
                plt.tight_layout()
                result["figures"]["variance_explained"] = fig_variance
            except Exception as e:
                result["message"] += f"Variance plot failed: {str(e)[:50]}... "
            
            # 2. PCA scatter plot (first two components)
            if n_components >= 2:
                try:
                    fig_scatter, ax = plt.subplots(figsize=(10, 8))
                    
                    if target and target in df.columns:
                        # Align target with cleaned data
                        target_values = df.loc[X.index, target]
                        target_clean = target_values.dropna()
                        
                        # Ensure alignment
                        common_idx = X.index.intersection(target_clean.index)
                        if len(common_idx) > 0:
                            X_pca_plot = X_pca[:len(common_idx)]
                            target_plot = target_clean[common_idx]
                            
                            if pd.api.types.is_numeric_dtype(target_plot):
                                scatter = ax.scatter(X_pca_plot[:, 0], X_pca_plot[:, 1], 
                                                   c=target_plot, cmap='viridis', alpha=0.6)
                                plt.colorbar(scatter, label=target)
                            else:
                                unique_targets = target_plot.unique()
                                colors = plt.cm.Set1(np.linspace(0, 1, min(len(unique_targets), 10)))
                                for i, target_val in enumerate(unique_targets[:10]):  # Limit colors
                                    mask = target_plot == target_val
                                    ax.scatter(X_pca_plot[mask, 0], X_pca_plot[mask, 1], 
                                              c=[colors[i]], label=str(target_val)[:20], alpha=0.6)
                                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
                    
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    ax.set_title('PCA: First Two Principal Components')
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    result["figures"]["pca_scatter"] = fig_scatter
                except Exception as e:
                    result["message"] += f"Scatter plot failed: {str(e)[:50]}... "
            
            # 3. Feature importance in components (loadings)
            if advanced_options.get('show_loadings', True):
                try:
                    n_components_to_show = min(4, n_components)
                    fig_loadings, ax = plt.subplots(figsize=(12, max(8, len(non_constant_cols) * 0.3)))
                    
                    components_df = pd.DataFrame(
                        pca.components_[:n_components_to_show].T,
                        columns=[f'PC{i+1}' for i in range(n_components_to_show)],
                        index=non_constant_cols
                    )
                    
                    sns.heatmap(components_df, annot=True, cmap='RdBu_r', 
                              center=0, fmt='.2f', cbar_kws={'label': 'Loading'}, ax=ax)
                    ax.set_title('Feature Loadings on Principal Components')
                    ax.set_ylabel('Original Features')
                    ax.set_xlabel('Principal Components')
                    plt.tight_layout()
                    result["figures"]["loadings"] = fig_loadings
                except Exception as e:
                    result["message"] += f"Loadings plot failed: {str(e)[:50]}... "
            
            # Generate insights
            try:
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                components_for_80_var = np.where(cumvar >= 0.8)[0]
                components_for_95_var = np.where(cumvar >= 0.95)[0]
                
                components_for_80_var = components_for_80_var[0] + 1 if len(components_for_80_var) > 0 else n_components
                components_for_95_var = components_for_95_var[0] + 1 if len(components_for_95_var) > 0 else n_components
                
                # Top contributing features
                top_features = {}
                for i in range(min(3, n_components)):
                    abs_loadings = np.abs(pca.components_[i])
                    top_idx = np.argmax(abs_loadings)
                    top_features[f'PC{i+1}'] = {
                        'feature': non_constant_cols[top_idx],
                        'loading': float(pca.components_[i][top_idx])
                    }
                
                result["insights"] = {
                    'total_variance_explained': float(pca.explained_variance_ratio_.sum()),
                    'components_for_80_percent': int(components_for_80_var),
                    'components_for_95_percent': int(components_for_95_var),
                    'dimensionality_reduction': {
                        'original_features': len(non_constant_cols),
                        'recommended_components': int(components_for_80_var),
                        'variance_preserved': float(cumvar[components_for_80_var-1] if components_for_80_var <= len(cumvar) else cumvar[-1])
                    },
                    'top_contributing_features': top_features
                }
            except Exception as e:
                result["insights"] = {
                    'error': f"Insights generation failed: {str(e)}",
                    'total_variance_explained': float(pca.explained_variance_ratio_.sum()) if 'pca' in locals() else 0
                }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in PCA analysis: {str(e)}"
        
        return result
    
    def create_umap_analysis(self, df: pd.DataFrame, target: str = None,
                           n_neighbors: int = 15, min_dist: float = 0.1,
                           metric: str = 'euclidean', n_components: int = 2,
                           advanced_options: Dict = None) -> Dict[str, Any]:
        """Perform UMAP analysis for nonlinear dimensionality reduction."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        if advanced_options is None:
            advanced_options = {}
        
        try:
            # Get robust numeric columns
            numeric_cols = self._get_robust_numeric_columns(df)
            if target and target in numeric_cols:
                numeric_cols.remove(target)
            
            if len(numeric_cols) < 2:
                result["status"] = "error"
                result["message"] = f"Need at least 2 numeric features for UMAP. Found {len(numeric_cols)} valid numeric columns."
                return result
            
            # Prepare data with robust handling
            X = df[numeric_cols].copy()
            
            # Convert to numeric and handle missing values
            for col in numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Remove rows with too many missing values
            X = X.dropna(thresh=len(numeric_cols) * 0.5)
            
            if len(X) < 10:
                result["status"] = "error"
                result["message"] = f"Need at least 10 complete samples for UMAP. Found {len(X)} after cleaning."
                return result
            
            # Fill remaining missing values
            X = X.fillna(X.median())
            
            # Check for constant columns and remove them
            non_constant_cols = []
            for col in numeric_cols:
                if X[col].var() > 1e-10:
                    non_constant_cols.append(col)
            
            if len(non_constant_cols) < 2:
                result["status"] = "error"
                result["message"] = f"Need at least 2 non-constant features. Found {len(non_constant_cols)}."
                return result
            
            X = X[non_constant_cols]
            
            # Scale data
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"Data scaling failed: {str(e)}"
                return result
            
            # Adjust parameters based on data size
            actual_n_neighbors = min(n_neighbors, len(X) - 1)
            if actual_n_neighbors < 2:
                actual_n_neighbors = min(5, len(X) - 1)
            
            # Fit UMAP with error handling
            try:
                umap_model = umap.UMAP(
                n_neighbors=actual_n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_components=n_components,
                random_state=42,
                n_jobs=1
                )
                
                X_umap = umap_model.fit_transform(X_scaled)
                
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"UMAP fitting failed: {str(e)}"
                return result
            
            # Create visualization for 2D case
            if n_components == 2:
                try:
                    fig_umap, ax = plt.subplots(figsize=(12, 8))
                    
                    target_values = None
                    if target and target in df.columns:
                        # Align target with cleaned data
                        target_values = df.loc[X.index, target]
                        
                        # Handle different target types
                        if pd.api.types.is_numeric_dtype(target_values):
                            # Remove NaN values from target
                            valid_mask = ~target_values.isna()
                            if valid_mask.sum() > 0:
                                scatter = ax.scatter(X_umap[valid_mask, 0], X_umap[valid_mask, 1], 
                                                   c=target_values[valid_mask], cmap='viridis', 
                                                   alpha=0.6, s=50)
                                plt.colorbar(scatter, label=target)
                        else:
                            # Categorical target
                            target_clean = target_values.dropna()
                            unique_targets = target_clean.unique()
                            
                            # Limit number of categories for visualization
                            if len(unique_targets) > 20:
                                top_categories = target_clean.value_counts().head(20).index
                                target_mask = target_clean.isin(top_categories)
                                target_clean = target_clean[target_mask]
                                X_umap_plot = X_umap[target_mask]
                                result["message"] += "Limited to top 20 categories for visualization. "
                            else:
                                X_umap_plot = X_umap
                            
                            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_targets)))
                            
                            for i, target_val in enumerate(unique_targets):
                                if len(unique_targets) <= 20:
                                    mask = target_clean == target_val
                                    if mask.sum() > 0:
                                        ax.scatter(X_umap_plot[mask, 0], X_umap_plot[mask, 1], 
                                                  c=[colors[i]], label=str(target_val)[:15], 
                                                  alpha=0.6, s=50)
                            
                            if len(unique_targets) <= 10:  # Only show legend for reasonable number of categories
                                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=50, c='blue')
                    
                    ax.set_xlabel('UMAP Component 1')
                    ax.set_ylabel('UMAP Component 2')
                    ax.set_title(f'UMAP Projection (neighbors={actual_n_neighbors}, min_dist={min_dist})')
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    result["figures"]["umap_2d"] = fig_umap
                    
                except Exception as e:
                    result["message"] += f"Visualization failed: {str(e)[:50]}... "
            
            # Generate insights
            try:
                cluster_analysis = self._analyze_umap_clusters(X_umap, target_values if target and target in df.columns else None)
                
                result["insights"] = {
                    'n_samples_used': int(len(X)),
                    'n_features_used': int(len(non_constant_cols)),
                    'features_used': non_constant_cols,
                    'parameters': {
                        'n_neighbors': int(actual_n_neighbors),
                        'min_dist': float(min_dist),
                        'metric': metric,
                        'n_components': int(n_components)
                    },
                    'cluster_analysis': cluster_analysis,
                    'data_quality': {
                        'missing_data_handled': True,
                        'scaling_applied': True,
                        'constant_features_removed': len(numeric_cols) - len(non_constant_cols)
                    }
                }
                
            except Exception as e:
                result["insights"] = {
                    'error': f"Insights generation failed: {str(e)}",
                    'n_samples_used': int(len(X)) if 'X' in locals() else 0,
                    'n_features_used': int(len(non_constant_cols)) if 'non_constant_cols' in locals() else 0
                }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in UMAP analysis: {str(e)}"
        
        return result
    
    def _analyze_umap_clusters(self, X_umap: np.ndarray, target_values: pd.Series = None) -> Dict:
        """Analyze clusters formed in UMAP space."""
        from sklearn.cluster import DBSCAN
        
        try:
            # Simple clustering analysis
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_umap)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            n_noise = list(clustering.labels_).count(-1)
            
            analysis = {
                'estimated_clusters': n_clusters,
                'noise_points': n_noise,
                'silhouette_score': None
            }
            
            if n_clusters > 1:
                from sklearn.metrics import silhouette_score
                analysis['silhouette_score'] = silhouette_score(X_umap, clustering.labels_)
            
            return analysis
        except:
            return {'estimated_clusters': 'Unable to analyze', 'noise_points': 0}
    
    def create_mutual_info_analysis(self, df: pd.DataFrame, target: str,
                                  feature_types: str = 'all', max_features: int = 20,
                                  advanced_options: Dict = None) -> Dict[str, Any]:
        """Calculate and visualize mutual information between features and target."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        try:
            if target not in df.columns:
                result["status"] = "error"
                result["message"] = f"Target column '{target}' not found"
                return result
            
            # Prepare features based on type selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if target in numeric_cols:
                numeric_cols.remove(target)
            if target in categorical_cols:
                categorical_cols.remove(target)
            
            if feature_types == 'numeric':
                feature_cols = numeric_cols
            elif feature_types == 'categorical':
                feature_cols = categorical_cols
            else:  # 'all'
                feature_cols = numeric_cols + categorical_cols
            
            if len(feature_cols) == 0:
                result["status"] = "error"
                result["message"] = f"No {feature_types} features found"
                return result
            
            # Limit number of features
            if len(feature_cols) > max_features:
                # Select top features by variance for numeric, by unique values for categorical
                if feature_types == 'numeric' or feature_types == 'all':
                    numeric_vars = df[numeric_cols].var().sort_values(ascending=False)
                    top_numeric = numeric_vars.head(max_features//2 if feature_types == 'all' else max_features).index.tolist()
                else:
                    top_numeric = []
                
                if feature_types == 'categorical' or feature_types == 'all':
                    categorical_nunique = df[categorical_cols].nunique().sort_values(ascending=False)
                    top_categorical = categorical_nunique.head(max_features//2 if feature_types == 'all' else max_features).index.tolist()
                else:
                    top_categorical = []
                
                feature_cols = top_numeric + top_categorical
                result["message"] += f"Limited to top {len(feature_cols)} features for performance. "
            
            # Prepare data
            analysis_df = df[feature_cols + [target]].dropna()
            if len(analysis_df) == 0:
                result["status"] = "error"
                result["message"] = "No complete cases found"
                return result
            
            # Calculate mutual information
            X = analysis_df[feature_cols]
            y = analysis_df[target]
            
            # Encode categorical variables for MI calculation
            X_encoded = X.copy()
            for col in categorical_cols:
                if col in X_encoded.columns:
                    X_encoded[col] = pd.Categorical(X_encoded[col]).codes
            
            # Choose MI function based on target type
            if pd.api.types.is_numeric_dtype(y):
                mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
            else:
                y_encoded = pd.Categorical(y).codes
                mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=42)
            
            # Create results DataFrame
            mi_df = pd.DataFrame({
                'Feature': feature_cols,
                'Mutual_Information': mi_scores,
                'Type': ['Numeric' if col in numeric_cols else 'Categorical' for col in feature_cols]
            }).sort_values('Mutual_Information', ascending=False)
            
            # Create visualization
            fig_mi = plt.figure(figsize=(12, max(6, len(feature_cols) * 0.3)))
            
            # Color by feature type
            colors = ['skyblue' if t == 'Numeric' else 'lightcoral' for t in mi_df['Type']]
            
            plt.barh(range(len(mi_df)), mi_df['Mutual_Information'], color=colors)
            plt.yticks(range(len(mi_df)), mi_df['Feature'])
            plt.xlabel('Mutual Information Score')
            plt.title(f'Mutual Information with Target: {target}')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (idx, row) in enumerate(mi_df.iterrows()):
                plt.text(row['Mutual_Information'] + 0.001, i, f"{row['Mutual_Information']:.3f}", 
                        va='center', ha='left', fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='skyblue', label='Numeric'),
                             Patch(facecolor='lightcoral', label='Categorical')]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            result["figures"]["mutual_info"] = fig_mi
            
            # Generate insights
            top_features = mi_df.head(5)
            result["insights"] = {
                'top_features': top_features.to_dict('records'),
                'average_mi_score': mi_scores.mean(),
                'max_mi_score': mi_scores.max(),
                'features_analyzed': len(feature_cols),
                'recommendations': self._generate_mi_recommendations(mi_df, target)
            }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in mutual information analysis: {str(e)}"
        
        return result
    
    def _generate_mi_recommendations(self, mi_df: pd.DataFrame, target: str) -> List[str]:
        """Generate recommendations based on mutual information analysis."""
        recommendations = []
        
        high_mi_features = mi_df[mi_df['Mutual_Information'] > mi_df['Mutual_Information'].quantile(0.8)]
        low_mi_features = mi_df[mi_df['Mutual_Information'] < mi_df['Mutual_Information'].quantile(0.2)]
        
        if len(high_mi_features) > 0:
            recommendations.append(
                f"Strong predictors found: {', '.join(high_mi_features['Feature'].head(3).tolist())} "
                f"show high mutual information with {target}. Consider these as primary features."
            )
        
        if len(low_mi_features) > 5:
            recommendations.append(
                f"Consider removing {len(low_mi_features)} features with very low mutual information "
                f"(< {mi_df['Mutual_Information'].quantile(0.2):.3f}) to reduce noise and overfitting."
            )
        
        # Check for mixed feature types performance
        numeric_avg = mi_df[mi_df['Type'] == 'Numeric']['Mutual_Information'].mean()
        categorical_avg = mi_df[mi_df['Type'] == 'Categorical']['Mutual_Information'].mean()
        
        if not pd.isna(numeric_avg) and not pd.isna(categorical_avg):
            if numeric_avg > categorical_avg * 1.5:
                recommendations.append(
                    "Numeric features show stronger relationships with target. "
                    "Consider feature engineering on categorical variables."
                )
            elif categorical_avg > numeric_avg * 1.5:
                recommendations.append(
                    "Categorical features dominate. Consider encoding techniques like "
                    "target encoding or embedding for better numeric representation."
                )
        
        return recommendations
    
    def create_feature_interaction_analysis(self, df: pd.DataFrame, target: str = None,
                                          interaction_method: str = 'correlation',
                                          max_interactions: int = 15,
                                          advanced_options: Dict = None) -> Dict[str, Any]:
        """Analyze feature interactions and their relationship with target."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        try:
            # Get numeric columns only and ensure they are actually numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target and target in numeric_cols:
                numeric_cols.remove(target)
            
            # Additional check: ensure columns have numeric data (not mixed types)
            truly_numeric_cols = []
            for col in numeric_cols:
                try:
                    # Test if column can be converted to numeric successfully
                    pd.to_numeric(df[col], errors='raise')
                    truly_numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue
            
            numeric_cols = truly_numeric_cols
            
            if not target:
                result["status"] = "error"
                result["message"] = "Target column is required for interaction analysis"
                return result
            
            if len(numeric_cols) < 2:
                result["status"] = "error"
                result["message"] = "Need at least 2 numeric features for interaction analysis"
                return result
            
            # Limit features if too many
            if len(numeric_cols) > 15:
                # Select features with highest variance
                feature_vars = df[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = feature_vars.head(15).index.tolist()
                result["message"] += f"Limited to top 15 features by variance for performance. "
            
            # Generate feature interactions
            interactions_data = []
            interaction_names = []
            
            # Create pairwise interactions
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    # Create interaction term
                    interaction_values = df[col1] * df[col2]
                    interaction_name = f"{col1} × {col2}"
                    
                    if target and target in df.columns:
                        # Calculate interaction strength with target
                        valid_mask = ~(pd.isna(interaction_values) | pd.isna(df[target]))
                        
                        if interaction_method == 'correlation' and pd.api.types.is_numeric_dtype(df[target]):
                            # Ensure both interaction and target are numeric
                            interaction_clean = pd.to_numeric(interaction_values[valid_mask], errors='coerce')
                            target_clean = pd.to_numeric(df[target][valid_mask], errors='coerce')
                            
                            # Remove any NaN values after conversion
                            final_mask = ~(pd.isna(interaction_clean) | pd.isna(target_clean))
                            
                            if final_mask.sum() > 1:  # Need at least 2 points for correlation
                                strength = abs(np.corrcoef(interaction_clean[final_mask], 
                                                         target_clean[final_mask])[0, 1])
                            else:
                                strength = 0.0
                        elif interaction_method == 'mutual_info':
                            # Ensure interaction values are numeric
                            interaction_clean = pd.to_numeric(interaction_values[valid_mask], errors='coerce')
                            target_clean = df[target][valid_mask]
                            
                            # Remove any remaining NaN values after conversion
                            final_mask = ~(pd.isna(interaction_clean) | pd.isna(target_clean))
                            
                            if final_mask.sum() > 0:
                                if pd.api.types.is_numeric_dtype(target_clean):
                                    target_final = pd.to_numeric(target_clean[final_mask], errors='coerce')
                                    strength = mutual_info_regression(
                                        interaction_clean[final_mask].values.reshape(-1, 1),
                                        target_final.values,
                                        random_state=42
                                    )[0]
                                else:
                                    target_encoded = pd.Categorical(target_clean[final_mask]).codes
                                    strength = mutual_info_classif(
                                        interaction_clean[final_mask].values.reshape(-1, 1),
                                        target_encoded,
                                        random_state=42
                                    )[0]
                            else:
                                strength = 0.0
                        else:
                            # Default to correlation method with proper type handling
                            interaction_clean = pd.to_numeric(interaction_values[valid_mask], errors='coerce')
                            target_clean = pd.to_numeric(df[target][valid_mask], errors='coerce')
                            
                            # Remove any NaN values after conversion
                            final_mask = ~(pd.isna(interaction_clean) | pd.isna(target_clean))
                            
                            if final_mask.sum() > 1:  # Need at least 2 points for correlation
                                strength = abs(np.corrcoef(interaction_clean[final_mask], 
                                                         target_clean[final_mask])[0, 1])
                            else:
                                strength = 0.0
                        
                        if not pd.isna(strength):
                            interactions_data.append({
                                'feature1': col1,
                                'feature2': col2,
                                'interaction_name': interaction_name,
                                'strength': strength,
                                'individual_corr1': abs(df[col1].corr(df[target])) if pd.api.types.is_numeric_dtype(df[target]) else 0,
                                'individual_corr2': abs(df[col2].corr(df[target])) if pd.api.types.is_numeric_dtype(df[target]) else 0
                            })
            
            if not interactions_data:
                result["status"] = "error"
                result["message"] = "No valid interactions could be computed"
                return result
            
            # Convert to DataFrame and sort by strength
            interactions_df = pd.DataFrame(interactions_data)
            interactions_df['interaction_gain'] = (
                interactions_df['strength'] - 
                interactions_df[['individual_corr1', 'individual_corr2']].max(axis=1)
            )
            interactions_df = interactions_df.sort_values('strength', ascending=False)
            
            # Limit to top interactions
            top_interactions = interactions_df.head(max_interactions)
            
            # 1. Interaction strength bar plot
            fig_interactions = plt.figure(figsize=(12, max(6, len(top_interactions) * 0.4)))
            
            y_pos = range(len(top_interactions))
            colors = ['green' if gain > 0 else 'orange' for gain in top_interactions['interaction_gain']]
            
            plt.barh(y_pos, top_interactions['strength'], color=colors, alpha=0.7)
            plt.yticks(y_pos, top_interactions['interaction_name'])
            plt.xlabel(f'{interaction_method.capitalize()} Strength with {target}')
            plt.title('Feature Interaction Strength Analysis')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (idx, row) in enumerate(top_interactions.iterrows()):
                plt.text(row['strength'] + 0.001, i, f"{row['strength']:.3f}", 
                        va='center', ha='left', fontsize=8)
            
            plt.tight_layout()
            result["figures"]["interaction_strength"] = fig_interactions
            
            # 2. Interaction vs Individual Features comparison
            if advanced_options and advanced_options.get('show_comparison', True):
                fig_comparison = plt.figure(figsize=(10, 6))
                
                x = range(len(top_interactions))
                width = 0.25
                
                plt.bar([i - width for i in x], top_interactions['individual_corr1'], 
                       width, label='Feature 1 Individual', alpha=0.7, color='lightblue')
                plt.bar(x, top_interactions['individual_corr2'], 
                       width, label='Feature 2 Individual', alpha=0.7, color='lightgreen')
                plt.bar([i + width for i in x], top_interactions['strength'], 
                       width, label='Interaction', alpha=0.7, color='coral')
                
                plt.xlabel('Feature Pairs')
                plt.ylabel('Strength')
                plt.title('Interaction vs Individual Feature Strength')
                plt.xticks(x, [name[:20] + '...' if len(name) > 20 else name 
                              for name in top_interactions['interaction_name']], rotation=45)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                result["figures"]["comparison"] = fig_comparison
            
            # 3. 3D Interaction Surface (for top interaction)
            if advanced_options and advanced_options.get('show_3d_surface', False) and len(top_interactions) > 0:
                top_interaction = top_interactions.iloc[0]
                feat1, feat2 = top_interaction['feature1'], top_interaction['feature2']
                
                # Create 3D surface plot
                fig_3d = plt.figure(figsize=(12, 8))
                ax = fig_3d.add_subplot(111, projection='3d')
                
                # Create meshgrid for surface
                x1_range = np.linspace(df[feat1].min(), df[feat1].max(), 20)
                x2_range = np.linspace(df[feat2].min(), df[feat2].max(), 20)
                X1, X2 = np.meshgrid(x1_range, x2_range)
                
                # Calculate interaction surface
                Z = X1 * X2
                
                # Plot surface
                surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
                
                # Add actual data points
                valid_mask = ~(pd.isna(df[feat1]) | pd.isna(df[feat2]))
                if target and target in df.columns:
                    scatter = ax.scatter(df[feat1][valid_mask], df[feat2][valid_mask], 
                                       df[feat1][valid_mask] * df[feat2][valid_mask],
                                       c=df[target][valid_mask], cmap='RdYlBu', s=20)
                    plt.colorbar(scatter, ax=ax, label=target)
                
                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)
                ax.set_zlabel(f'{feat1} × {feat2}')
                ax.set_title(f'3D Interaction Surface: {feat1} × {feat2}')
                
                result["figures"]["3d_surface"] = fig_3d
            
            # Generate insights
            positive_gain_interactions = interactions_df[interactions_df['interaction_gain'] > 0]
            
            result["insights"] = {
                'total_interactions_analyzed': len(interactions_df),
                'beneficial_interactions': len(positive_gain_interactions),
                'top_interaction': {
                    'features': f"{top_interactions.iloc[0]['feature1']} × {top_interactions.iloc[0]['feature2']}",
                    'strength': top_interactions.iloc[0]['strength'],
                    'gain_over_individual': top_interactions.iloc[0]['interaction_gain']
                },
                'average_interaction_strength': interactions_df['strength'].mean(),
                'recommendations': self._generate_interaction_recommendations(interactions_df, target)
            }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in interaction analysis: {str(e)}"
        
        return result
    
    def _generate_interaction_recommendations(self, interactions_df: pd.DataFrame, target: str) -> List[str]:
        """Generate recommendations based on interaction analysis."""
        recommendations = []
        
        # Strong interactions
        strong_interactions = interactions_df[interactions_df['strength'] > interactions_df['strength'].quantile(0.8)]
        if len(strong_interactions) > 0:
            recommendations.append(
                f"Found {len(strong_interactions)} strong feature interactions. "
                f"Consider creating interaction terms for modeling, especially: "
                f"{strong_interactions.iloc[0]['interaction_name']}"
            )
        
        # Beneficial interactions (better than individual features)
        beneficial = interactions_df[interactions_df['interaction_gain'] > 0.05]
        if len(beneficial) > 0:
            recommendations.append(
                f"{len(beneficial)} interactions show significant improvement over individual features. "
                f"These could be valuable engineered features."
            )
        
        # Non-beneficial interactions
        if len(interactions_df[interactions_df['interaction_gain'] < 0]) > len(interactions_df) * 0.7:
            recommendations.append(
                "Most interactions don't improve upon individual features. "
                "Focus on feature selection rather than interaction engineering."
            )
        
        return recommendations
    
    def create_advanced_correlation_analysis(self, df: pd.DataFrame, 
                                           correlation_method: str = 'pearson',
                                           cluster_method: str = 'ward',
                                           advanced_options: Dict = None) -> Dict[str, Any]:
        """Advanced correlation analysis with clustering and network analysis."""
        result = {"figures": {}, "status": "success", "message": "", "insights": {}}
        
        try:
            # Get numeric columns and ensure they are truly numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter out columns that aren't actually numeric or have constant values
            truly_numeric_cols = []
            for col in numeric_cols:
                try:
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    # Check if column has variance (not constant)
                    if len(col_data) > 1 and col_data.var() > 1e-10:
                        truly_numeric_cols.append(col)
                except:
                    continue
            
            numeric_cols = truly_numeric_cols
            
            if len(numeric_cols) < 3:
                result["status"] = "error"
                result["message"] = f"Need at least 3 numeric features for advanced correlation analysis. Found: {len(numeric_cols)}"
                return result
            
            # Prepare clean dataset for correlation analysis
            df_clean = df[numeric_cols].copy()
            
            # Convert all columns to numeric and handle missing values
            for col in numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove rows with too many missing values
            df_clean = df_clean.dropna(thresh=len(numeric_cols) * 0.5)  # Keep rows with at least 50% non-null values
            
            if len(df_clean) < 10:
                result["status"] = "error"
                result["message"] = "Insufficient data after cleaning for correlation analysis"
                return result
            
            # Fill remaining missing values with median
            df_clean = df_clean.fillna(df_clean.median())
            
            # Calculate correlation matrix with error handling
            try:
                if correlation_method == 'spearman':
                    corr_matrix = df_clean.corr(method='spearman')
                elif correlation_method == 'kendall':
                    corr_matrix = df_clean.corr(method='kendall')
                else:  # pearson
                    corr_matrix = df_clean.corr(method='pearson')
                
                # Check for NaN values in correlation matrix
                if corr_matrix.isnull().any().any():
                    result["status"] = "error"
                    result["message"] = "Correlation matrix contains NaN values. Check data quality."
                    return result
                    
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"Error calculating correlation matrix: {str(e)}"
                return result
            
            # Remove diagonal and duplicates for analysis
            corr_values = corr_matrix.values.copy()
            np.fill_diagonal(corr_values, 0)
            
            # 1. Clustered correlation heatmap
            if advanced_options and advanced_options.get('show_clustered_heatmap', True):
                try:
                    from scipy.cluster.hierarchy import linkage, dendrogram
                    from scipy.spatial.distance import squareform
                    
                    # Create distance matrix from correlation (ensure proper format)
                    distance_matrix = 1 - np.abs(corr_matrix.values)
                    
                    # Ensure diagonal is exactly zero
                    np.fill_diagonal(distance_matrix, 0.0)
                    
                    # Ensure matrix is symmetric
                    distance_matrix = (distance_matrix + distance_matrix.T) / 2
                    
                    # Clip negative values that might arise from floating point errors
                    distance_matrix = np.clip(distance_matrix, 0, 2)
                    
                    # Convert to condensed distance matrix
                    try:
                        condensed_distances = squareform(distance_matrix, checks=False)
                        
                        # Perform hierarchical clustering
                        linkage_matrix = linkage(condensed_distances, method=cluster_method)
                        
                        # Get dendrogram order
                        dendro = dendrogram(linkage_matrix, no_plot=True)
                        cluster_order = dendro['leaves']
                        
                        # Reorder correlation matrix
                        clustered_corr = corr_matrix.iloc[cluster_order, cluster_order]
                        
                        fig_clustered = plt.figure(figsize=(12, 10))
                        
                        # Plot clustered heatmap
                        mask = np.triu(np.ones_like(clustered_corr, dtype=bool))
                        
                        # Handle case where all correlations are very similar
                        vmin, vmax = clustered_corr.min().min(), clustered_corr.max().max()
                        if abs(vmax - vmin) < 0.01:  # Very small range
                            vmin, vmax = -1, 1
                        
                        sns.heatmap(clustered_corr, mask=mask, annot=True, cmap='RdBu_r', 
                                  center=0, square=True, fmt='.2f', 
                                  cbar_kws={'label': 'Correlation'}, vmin=vmin, vmax=vmax)
                        plt.title(f'Clustered Correlation Matrix ({correlation_method.capitalize()})')
                        plt.tight_layout()
                        result["figures"]["clustered_heatmap"] = fig_clustered
                        
                    except Exception as cluster_error:
                        # Fallback: regular heatmap without clustering
                        fig_clustered = plt.figure(figsize=(12, 10))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                                  center=0, square=True, fmt='.2f', 
                                  cbar_kws={'label': 'Correlation'})
                        plt.title(f'Correlation Matrix ({correlation_method.capitalize()}) - No Clustering')
                        plt.tight_layout()
                        result["figures"]["clustered_heatmap"] = fig_clustered
                        result["message"] += f"Clustering failed, showing regular heatmap. Error: {str(cluster_error)[:100]}... "
                        
                except ImportError:
                    result["message"] += "Scipy not available for clustering. "
                except Exception as e:
                    result["message"] += f"Clustering error: {str(e)[:100]}... "
            
            # 2. Correlation network analysis
            if advanced_options and advanced_options.get('show_network', True):
                try:
                    # Find significant correlations
                    threshold = advanced_options.get('correlation_threshold', 0.5)
                    significant_corrs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                                significant_corrs.append({
                                    'feature1': corr_matrix.columns[i],
                                    'feature2': corr_matrix.columns[j],
                                    'correlation': corr_val
                                })
                    
                    if significant_corrs:
                        # Create network visualization
                        fig_network = plt.figure(figsize=(12, 10))
                        
                        # Simple network layout (circular)
                        n_features = len(numeric_cols)
                        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
                        pos = {feat: (np.cos(angle), np.sin(angle)) 
                               for feat, angle in zip(numeric_cols, angles)}
                        
                        # Draw nodes
                        for feat, (x, y) in pos.items():
                            plt.scatter(x, y, s=500, c='lightblue', alpha=0.7, edgecolors='black')
                            # Adjust text position to avoid overlap
                            plt.text(x*1.15, y*1.15, feat[:10], ha='center', va='center', 
                                   fontsize=8, weight='bold')
                        
                        # Draw edges
                        for corr_info in significant_corrs:
                            feat1, feat2 = corr_info['feature1'], corr_info['feature2']
                            if feat1 in pos and feat2 in pos:
                                x1, y1 = pos[feat1]
                                x2, y2 = pos[feat2]
                                
                                # Color and width based on correlation strength
                                color = 'red' if corr_info['correlation'] > 0 else 'blue'
                                width = min(abs(corr_info['correlation']) * 4, 4)  # Cap width
                                alpha = min(abs(corr_info['correlation']) * 1.5, 0.8)
                                
                                plt.plot([x1, x2], [y1, y2], color=color, 
                                       linewidth=width, alpha=alpha)
                        
                        plt.xlim(-1.5, 1.5)
                        plt.ylim(-1.5, 1.5)
                        plt.title(f'Correlation Network (|r| >= {threshold})\nRed=Positive, Blue=Negative')
                        plt.axis('off')
                        result["figures"]["network"] = fig_network
                    else:
                        result["message"] += f"No correlations >= {threshold} found for network visualization. "
                        
                except Exception as e:
                    result["message"] += f"Network visualization error: {str(e)[:100]}... "
            
            # 3. Correlation distribution analysis
            try:
                fig_dist = plt.figure(figsize=(12, 5))
                
                # Get all correlation values (excluding diagonal)
                all_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val):
                            all_corrs.append(corr_val)
                
                if len(all_corrs) > 0:
                    plt.subplot(1, 2, 1)
                    plt.hist(all_corrs, bins=min(30, len(all_corrs)//2), alpha=0.7, 
                           color='skyblue', edgecolor='black')
                    plt.xlabel('Correlation Coefficient')
                    plt.ylabel('Frequency')
                    plt.title('Distribution of Correlations')
                    plt.grid(alpha=0.3)
                    
                    plt.subplot(1, 2, 2)
                    plt.boxplot(all_corrs)
                    plt.ylabel('Correlation Coefficient')
                    plt.title('Correlation Distribution Summary')
                    plt.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    result["figures"]["distribution"] = fig_dist
                else:
                    result["message"] += "No valid correlations found for distribution analysis. "
                    
            except Exception as e:
                result["message"] += f"Distribution analysis error: {str(e)[:100]}... "
            
            # Generate insights
            if len(all_corrs) > 0:
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                result["insights"] = {
                    'correlation_summary': {
                        'mean_absolute_correlation': np.mean(np.abs(all_corrs)),
                        'max_correlation': max(all_corrs) if all_corrs else 0,
                        'min_correlation': min(all_corrs) if all_corrs else 0,
                        'high_correlation_pairs': len(high_corr_pairs)
                    },
                    'multicollinearity_analysis': {
                        'features_with_high_correlation': len(set([pair[0] for pair in high_corr_pairs] + 
                                                                [pair[1] for pair in high_corr_pairs])),
                        'strongest_correlation': high_corr_pairs[0] if high_corr_pairs else None
                    },
                    'recommendations': self._generate_correlation_recommendations(all_corrs, high_corr_pairs)
                }
            else:
                result["insights"] = {
                    'correlation_summary': {'mean_absolute_correlation': 0, 'max_correlation': 0, 
                                          'min_correlation': 0, 'high_correlation_pairs': 0},
                    'multicollinearity_analysis': {'features_with_high_correlation': 0, 'strongest_correlation': None},
                    'recommendations': ["No valid correlations could be calculated. Check data quality."]
                }
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error in advanced correlation analysis: {str(e)}"
        
        return result
    
    def _generate_correlation_recommendations(self, all_corrs: List[float], 
                                            high_corr_pairs: List[Tuple]) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []
        
        mean_abs_corr = np.mean(np.abs(all_corrs))
        
        if mean_abs_corr > 0.5:
            recommendations.append(
                f"High average correlation ({mean_abs_corr:.2f}) detected. "
                "Consider dimensionality reduction techniques like PCA."
            )
        
        if len(high_corr_pairs) > 0:
            recommendations.append(
                f"Found {len(high_corr_pairs)} feature pairs with correlation > 0.8. "
                "This indicates multicollinearity - consider removing redundant features."
            )
            
            if len(high_corr_pairs) > 5:
                recommendations.append(
                    "Severe multicollinearity detected. Use regularization techniques "
                    "(Ridge, Lasso) or feature selection methods."
                )
        
        if mean_abs_corr < 0.2:
            recommendations.append(
                "Low correlations suggest features are mostly independent. "
                "This is good for avoiding multicollinearity but may indicate "
                "need for feature engineering or interaction terms."
            )
        
        return recommendations
    
    def create_comprehensive_report(self, df: pd.DataFrame, target: str = None,
                                  analyses_to_include: List[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive AI-powered analysis report."""
        if analyses_to_include is None:
            analyses_to_include = ['structure', 'missingness', 'pca', 'mutual_info', 'interactions']
        
        report = {
            'executive_summary': {},
            'detailed_analyses': {},
            'ai_recommendations': [],
            'action_items': [],
            'data_quality_score': 0
        }
        
        try:
            # 1. Data structure analysis
            if 'structure' in analyses_to_include:
                structure_analysis = self.analyze_data_structure(df, target)
                report['detailed_analyses']['structure'] = structure_analysis
            
            # 2. Missing data analysis
            if 'missingness' in analyses_to_include:
                missingness_analysis = self.create_missingness_analysis(df)
                report['detailed_analyses']['missingness'] = missingness_analysis
            
            # 3. PCA analysis
            if 'pca' in analyses_to_include and len(df.select_dtypes(include=[np.number]).columns) >= 2:
                pca_analysis = self.create_pca_analysis(df, target)
                report['detailed_analyses']['pca'] = pca_analysis
            
            # 4. Mutual information analysis
            if 'mutual_info' in analyses_to_include and target:
                mi_analysis = self.create_mutual_info_analysis(df, target)
                report['detailed_analyses']['mutual_info'] = mi_analysis
            
            # 5. Feature interactions
            if 'interactions' in analyses_to_include and target:
                interaction_analysis = self.create_feature_interaction_analysis(df, target)
                report['detailed_analyses']['interactions'] = interaction_analysis
            
            # Generate executive summary
            report['executive_summary'] = self._generate_executive_summary(df, report['detailed_analyses'])
            
            # Compile AI recommendations
            all_recommendations = []
            for analysis_name, analysis_result in report['detailed_analyses'].items():
                if isinstance(analysis_result, dict) and 'insights' in analysis_result:
                    insights = analysis_result['insights']
                    if 'recommendations' in insights:
                        if isinstance(insights['recommendations'], list):
                            all_recommendations.extend(insights['recommendations'])
                        else:
                            all_recommendations.append(str(insights['recommendations']))
            
            report['ai_recommendations'] = all_recommendations
            
            # Generate action items
            report['action_items'] = self._generate_action_items(report['detailed_analyses'])
            
            # Calculate data quality score
            report['data_quality_score'] = self._calculate_data_quality_score(df, report['detailed_analyses'])
            
        except Exception as e:
            report['error'] = f"Error generating comprehensive report: {str(e)}"
        
        return report
    
    def _generate_executive_summary(self, df: pd.DataFrame, analyses: Dict) -> Dict[str, Any]:
        """Generate executive summary from all analyses."""
        summary = {
            'dataset_overview': {
                'rows': len(df),
                'features': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'key_findings': [],
            'data_readiness': 'Unknown',
            'complexity_assessment': 'Medium'
        }
        
        # Extract key findings from each analysis
        if 'structure' in analyses:
            structure = analyses['structure']
            if 'complexity_score' in structure:
                if structure['complexity_score'] > 70:
                    summary['complexity_assessment'] = 'High'
                elif structure['complexity_score'] < 30:
                    summary['complexity_assessment'] = 'Low'
        
        if 'missingness' in analyses:
            missing_analysis = analyses['missingness']
            if 'insights' in missing_analysis:
                missing_features = missing_analysis['insights'].get('total_missing_features', 0)
                if missing_features > 0:
                    summary['key_findings'].append(
                        f"{missing_features} features have missing values requiring attention"
                    )
        
        if 'pca' in analyses:
            pca_analysis = analyses['pca']
            if 'insights' in pca_analysis:
                components_80 = pca_analysis['insights'].get('components_for_80_percent', 0)
                original_features = pca_analysis['insights'].get('dimensionality_reduction', {}).get('original_features', 0)
                if components_80 < original_features * 0.5:
                    summary['key_findings'].append(
                        f"High dimensionality: {components_80} components explain 80% of variance from {original_features} features"
                    )
        
        # Determine data readiness
        issues = len([finding for finding in summary['key_findings'] 
                     if any(word in finding.lower() for word in ['missing', 'error', 'issue', 'problem'])])
        
        if issues == 0:
            summary['data_readiness'] = 'Ready for modeling'
        elif issues <= 2:
            summary['data_readiness'] = 'Minor preprocessing needed'
        else:
            summary['data_readiness'] = 'Significant preprocessing required'
        
        return summary
    
    def _generate_action_items(self, analyses: Dict) -> List[Dict[str, str]]:
        """Generate prioritized action items based on analyses."""
        action_items = []
        
        # High priority items from structure analysis
        if 'structure' in analyses:
            structure = analyses['structure']
            if 'ai_recommendations' in structure:
                for rec in structure['ai_recommendations']:
                    if rec['priority'] == 'high':
                        action_items.append({
                            'priority': 'High',
                            'category': rec['type'],
                            'action': rec['action'],
                            'rationale': rec['description']
                        })
        
        # Medium priority items from other analyses
        for analysis_name, analysis_result in analyses.items():
            if isinstance(analysis_result, dict) and 'insights' in analysis_result:
                insights = analysis_result['insights']
                if 'recommendations' in insights and isinstance(insights['recommendations'], list):
                    for rec in insights['recommendations'][:2]:  # Top 2 recommendations
                        action_items.append({
                            'priority': 'Medium',
                            'category': f'{analysis_name}_optimization',
                            'action': rec,
                            'rationale': f'Based on {analysis_name} analysis'
                        })
        
        return action_items[:10]  # Limit to top 10 action items
    
    def _calculate_data_quality_score(self, df: pd.DataFrame, analyses: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        score_components = []
        
        # Completeness score (40% weight)
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        completeness_score = max(0, 100 - missing_pct * 2)  # Penalize missing data
        score_components.append(completeness_score * 0.4)
        
        # Consistency score (30% weight) - based on data types and duplicates
        duplicate_pct = df.duplicated().sum() / len(df) * 100
        consistency_score = max(0, 100 - duplicate_pct * 5)  # Penalize duplicates
        score_components.append(consistency_score * 0.3)
        
        # Usefulness score (30% weight) - based on variance and correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for low variance features
            low_variance_count = sum(1 for col in numeric_cols if df[col].var() < 1e-6)
            low_variance_pct = low_variance_count / len(numeric_cols) * 100
            usefulness_score = max(0, 100 - low_variance_pct * 10)
        else:
            usefulness_score = 50  # Neutral score for non-numeric data
        
        score_components.append(usefulness_score * 0.3)
        
        return sum(score_components)
from scipy.stats import kruskal
from sklearn.preprocessing import LabelEncoder

import traceback
from collections import Counter

class RobustAutomatedInsightGenerator:
    """
    Highly Robust AI-Powered Insight Generator
    Handles all data types, edge cases, and generates advanced insights
    """
    
    def __init__(self):
        self.insights = {
            'data_composition': {},
            'target_insights': {},
            'feature_patterns': {},
            'statistical_warnings': {},
            'data_quality': {},
            'relationships': {},
            'key_insights': [],
            'ai_recommendations': {},
            'business_insights': []
        }
        self.df = None
        self.target_column = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.boolean_cols = []
        
    def generate_insights(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Main function to generate all insights with robust error handling
        """
        try:
            # Validate input
            if df is None or df.empty:
                return self._create_error_response("Empty or None DataFrame provided")
            
            if df.shape[0] < 2:
                return self._create_error_response("Dataset must have at least 2 rows")
                
            # Initialize
            self.df = df.copy()
            self.target_column = target_column
            self._reset_insights()
            
            # Robust column type detection
            self._detect_column_types()
            
            # Generate insights with individual error handling
            self._safe_analyze_data_composition()
            self._safe_analyze_target_variable()
            self._safe_analyze_feature_patterns()
            self._safe_generate_statistical_warnings()
            self._safe_analyze_data_quality()
            self._safe_discover_relationships()
            self._safe_generate_ai_recommendations()
            self._safe_generate_business_insights()
            self._safe_generate_key_insights()
            
            return self.insights
            
        except Exception as e:
            return self._create_error_response(f"Critical error in insight generation: {str(e)}")
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            'error': True,
            'message': error_msg,
            'status': 'failed',
            'insights': {},
            'recommendations': ['Fix data issues and try again', 'Check data format and types']
        }
    
    def _reset_insights(self):
        """Reset insights dictionary"""
        self.insights = {
            'data_composition': {},
            'target_insights': {},
            'feature_patterns': {},
            'statistical_warnings': {},
            'data_quality': {},
            'relationships': {},
            'key_insights': [],
            'ai_recommendations': {},
            'business_insights': []
        }
    
    def _detect_column_types(self):
        """Robust column type detection"""
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.boolean_cols = []
        
        for col in self.df.columns:
            try:
                # Skip target column from feature lists
                if col == self.target_column:
                    continue
                    
                col_data = self.df[col].dropna()
                
                # Skip empty columns
                if len(col_data) == 0:
                    continue
                
                # Detect datetime
                if self._is_datetime_column(col_data):
                    self.datetime_cols.append(col)
                # Detect boolean
                elif self._is_boolean_column(col_data):
                    self.boolean_cols.append(col)
                # Detect numeric
                elif self._is_numeric_column(col_data):
                    self.numeric_cols.append(col)
                # Everything else is categorical
                else:
                    self.categorical_cols.append(col)
                    
            except Exception as e:
                # If column detection fails, treat as categorical
                if col != self.target_column:
                    self.categorical_cols.append(col)
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
    
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            try:
                sample = series.dropna().head(100) 
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
    
                return parsed.notna().mean() > 0.8
            except:
                return False

    return False

    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if column contains boolean data"""
        try:
            unique_vals = series.unique()
            
            # Check for explicit boolean
            if series.dtype == bool:
                return True
            
            # Check for binary values
            if len(unique_vals) == 2:
                # Common boolean patterns
                bool_patterns = [
                    {'True', 'False'}, {'true', 'false'}, {'TRUE', 'FALSE'},
                    {'Yes', 'No'}, {'yes', 'no'}, {'YES', 'NO'},
                    {'Y', 'N'}, {'y', 'n'}, {1, 0}, {'1', '0'},
                    {'Male', 'Female'}, {'M', 'F'}, {'Active', 'Inactive'}
                ]
                
                unique_set = set(str(v).strip() for v in unique_vals)
                return unique_set in bool_patterns
            
            return False
        except:
            return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if column contains numeric data"""
        try:
            # Check if already numeric
            if pd.api.types.is_numeric_dtype(series):
                return True
            
            # Try to convert to numeric
            pd.to_numeric(series, errors='raise')
            return True
        except:
            return False
    
    def _safe_analyze_data_composition(self):
        """Safely analyze data composition"""
        try:
            n_rows, n_cols = self.df.shape
            
            # Missing values analysis
            missing_info = self._analyze_missing_values()
            
            # Duplicate analysis
            duplicate_info = self._analyze_duplicates()
            
            # Memory usage
            memory_info = self._analyze_memory_usage()
            
            self.insights['data_composition'] = {
                'shape': {'rows': n_rows, 'columns': n_cols},
                'feature_types': {
                    'numerical': len(self.numeric_cols),
                    'categorical': len(self.categorical_cols),
                    'datetime': len(self.datetime_cols),
                    'boolean': len(self.boolean_cols),
                    'numerical_features': self.numeric_cols,
                    'categorical_features': self.categorical_cols,
                    'datetime_features': self.datetime_cols,
                    'boolean_features': self.boolean_cols
                },
                'missing_data': missing_info,
                'duplicates': duplicate_info,
                'memory_usage': memory_info,
                'data_density': self._calculate_data_density()
            }
            
        except Exception as e:
            self.insights['data_composition'] = {'error': f"Failed to analyze data composition: {str(e)}"}
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Detailed missing value analysis"""
        try:
            missing_counts = self.df.isnull().sum()
            total_cells = self.df.shape[0] * self.df.shape[1]
            total_missing = missing_counts.sum()
            
            # Per column missing analysis
            missing_by_column = {}
            for col in self.df.columns:
                missing_count = missing_counts[col]
                missing_pct = (missing_count / self.df.shape[0]) * 100
                missing_by_column[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
            
            # Missing patterns
            missing_patterns = self._find_missing_patterns()
            
            return {
                'total_missing': int(total_missing),
                'missing_percentage': round((total_missing / total_cells) * 100, 2),
                'by_column': missing_by_column,
                'patterns': missing_patterns,
                'completely_missing_columns': [col for col, info in missing_by_column.items() if info['percentage'] == 100],
                'high_missing_columns': [col for col, info in missing_by_column.items() if info['percentage'] > 50]
            }
        except Exception as e:
            return {'error': f"Missing value analysis failed: {str(e)}"}
    
    def _find_missing_patterns(self) -> List[str]:
        """Find patterns in missing data"""
        patterns = []
        try:
            # Check for columns with similar missing patterns
            missing_df = self.df.isnull()
            
            # Find columns that are missing together
            for col1 in self.df.columns:
                for col2 in self.df.columns:
                    if col1 != col2:
                        # Check if they're missing together often
                        both_missing = (missing_df[col1] & missing_df[col2]).sum()
                        col1_missing = missing_df[col1].sum()
                        
                        if col1_missing > 0 and both_missing / col1_missing > 0.8:
                            patterns.append(f"'{col1}' and '{col2}' are often missing together")
            
            return patterns[:5]  # Limit to top 5 patterns
        except:
            return []
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate data"""
        try:
            # Full row duplicates
            full_duplicates = self.df.duplicated().sum()
            full_dup_pct = (full_duplicates / self.df.shape[0]) * 100
            
            # Subset duplicates (excluding ID-like columns)
            potential_id_cols = [col for col in self.df.columns if 
                                any(keyword in col.lower() for keyword in ['id', 'key', 'index', 'uuid'])]
            
            subset_cols = [col for col in self.df.columns if col not in potential_id_cols]
            subset_duplicates = 0
            if subset_cols:
                subset_duplicates = self.df[subset_cols].duplicated().sum()
            
            return {
                'full_duplicates': int(full_duplicates),
                'full_duplicate_percentage': round(full_dup_pct, 2),
                'subset_duplicates': int(subset_duplicates),
                'potential_id_columns': potential_id_cols
            }
        except Exception as e:
            return {'error': f"Duplicate analysis failed: {str(e)}"}
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage"""
        try:
            memory_usage = self.df.memory_usage(deep=True)
            total_memory = memory_usage.sum()
            
            # Memory by column type
            memory_by_type = {}
            for col in self.df.columns:
                col_memory = memory_usage[col]
                if col in self.numeric_cols:
                    memory_by_type['numeric'] = memory_by_type.get('numeric', 0) + col_memory
                elif col in self.categorical_cols:
                    memory_by_type['categorical'] = memory_by_type.get('categorical', 0) + col_memory
                elif col in self.datetime_cols:
                    memory_by_type['datetime'] = memory_by_type.get('datetime', 0) + col_memory
                else:
                    memory_by_type['other'] = memory_by_type.get('other', 0) + col_memory
            
            return {
                'total_memory_mb': round(total_memory / (1024 * 1024), 2),
                'memory_by_type_mb': {k: round(v / (1024 * 1024), 2) for k, v in memory_by_type.items()},
                'largest_columns': dict(memory_usage.nlargest(5))
            }
        except Exception as e:
            return {'error': f"Memory analysis failed: {str(e)}"}
    
    def _calculate_data_density(self) -> float:
        """Calculate data density (non-null percentage)"""
        try:
            total_cells = self.df.shape[0] * self.df.shape[1]
            non_null_cells = total_cells - self.df.isnull().sum().sum()
            return round((non_null_cells / total_cells) * 100, 2)
        except:
            return 0.0
    
    def _safe_analyze_target_variable(self):
        """Safely analyze target variable"""
        try:
            if not self.target_column or self.target_column not in self.df.columns:
                self.insights['target_insights'] = {'message': 'No target variable specified or column not found'}
                return
            
            target = self.df[self.target_column]
            target_analysis = self._comprehensive_target_analysis(target)
            self.insights['target_insights'] = target_analysis
            
        except Exception as e:
            self.insights['target_insights'] = {'error': f"Target analysis failed: {str(e)}"}
    
    def _comprehensive_target_analysis(self, target: pd.Series) -> Dict[str, Any]:
        """Comprehensive target variable analysis"""
        try:
            total_count = len(target)
            unique_count = target.nunique()
            unique_ratio = unique_count / total_count
            dtype = str(target.dtype)

            analysis = {
                'column_name': self.target_column,
                'data_type': dtype,
                'missing_count': int(target.isnull().sum()),
                'missing_percentage': round((target.isnull().sum() / total_count) * 100, 2),
                'unique_values': int(unique_count),
                'unique_ratio': round(unique_ratio, 4)
            }

            clean_target = target.dropna()
            if len(clean_target) == 0:
                analysis['error'] = 'Target column is completely empty'
                return analysis

            # Step 1: Intelligent task detection
            task_type, task_info = self._determine_task_type(clean_target)

            # Step 2: Manual override for common autoML misclassifications
            if pd.api.types.is_numeric_dtype(clean_target):
                if unique_count > 50 and unique_ratio > 0.3:
                    # Likely continuous
                    if task_type != 'regression':
                        task_type = 'regression'
                        task_info = {'reason': 'Overriding: high unique numeric values suggest regression'}
            
            analysis['task_type'] = task_type
            analysis['task_info'] = task_info

            # Step 3: Type-specific deep dive
            if task_type == 'classification':
                analysis.update(self._analyze_classification_target(clean_target))
            elif task_type == 'regression':
                analysis.update(self._analyze_regression_target(clean_target))
            else:
                analysis['analysis'] = 'Unable to determine appropriate analysis type'

            # Step 4: Target-feature correlation
            analysis['top_correlated_features'] = self._find_robust_target_correlations(clean_target)

            return analysis

        except Exception as e:
            return {'error': f"Comprehensive target analysis failed: {str(e)}"}
    
    def _determine_task_type(self, target: pd.Series) -> Tuple[str, Dict]:
        """Intelligently determine if target is for classification or regression"""
        try:
            unique_count = target.nunique()
            total_count = len(target)
            unique_ratio = unique_count / total_count
            dtype = target.dtype
            reason = {}

            # Handle missing or too-small targets
            if total_count < 10:
                return 'unknown', {'reason': 'Too few samples to determine'}

            if pd.api.types.is_numeric_dtype(target):
                # Float dtype is almost always regression
                if pd.api.types.is_float_dtype(target):
                    return 'regression', {'reason': 'Float dtype with continuous values'}

                # Int dtype – look at unique values
                if pd.api.types.is_integer_dtype(target):
                    if unique_count <= 20 and unique_ratio < 0.1:
                        return 'classification', {'reason': f'Integer dtype with few unique values ({unique_count})'}
                    if unique_count > 50 and unique_ratio > 0.3:
                        return 'regression', {'reason': f'High unique values ({unique_count}) among integers'}
                    else:
                        return 'classification', {'reason': 'Integer dtype with moderate unique values'}
            
            # Categorical/object is classification
            if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
                return 'classification', {'reason': 'Non-numeric (object/categorical) target'}
            
            return 'unknown', {'reason': 'Unrecognized dtype or heuristic failed'}

        except Exception as e:
            return 'unknown', {'error': str(e)}
        
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target"""
        try:
            value_counts = target.value_counts()
            proportions = value_counts / len(target)
            
            # Class balance analysis
            balance_analysis = self._detailed_class_balance_analysis(value_counts, proportions)
            
            # Distribution analysis
            distribution = {
                'class_counts': value_counts.to_dict(),
                'class_proportions': {k: round(v, 4) for k, v in proportions.items()},
                'entropy': self._calculate_entropy(proportions),
                'gini_impurity': self._calculate_gini_impurity(proportions)
            }
            
            return {
                'class_balance': balance_analysis,
                'distribution': distribution,
                'recommendations': self._get_classification_recommendations(balance_analysis)
            }
            
        except Exception as e:
            return {'error': f"Classification analysis failed: {str(e)}"}
    
    def _detailed_class_balance_analysis(self, value_counts: pd.Series, proportions: pd.Series) -> Dict[str, Any]:
        """Detailed class balance analysis"""
        try:
            n_classes = len(value_counts)
            
            if n_classes < 2:
                return {
                    'status': 'single_class',
                    'message': 'Only one class present - cannot perform classification',
                    'severity': 'critical'
                }
            
            # Calculate balance metrics
            min_prop = proportions.min()
            max_prop = proportions.max()
            imbalance_ratio = max_prop / min_prop
            
            # Determine balance status
            if imbalance_ratio > 10:
                status = 'severely_imbalanced'
                severity = 'high'
            elif imbalance_ratio > 3:
                status = 'moderately_imbalanced'
                severity = 'medium'
            else:
                status = 'balanced'
                severity = 'low'
            
            # Detailed class information
            class_info = []
            for class_val, count in value_counts.items():
                class_info.append({
                    'class': str(class_val),
                    'count': int(count),
                    'percentage': round(proportions[class_val] * 100, 2)
                })
            
            return {
                'status': status,
                'severity': severity,
                'n_classes': n_classes,
                'imbalance_ratio': round(imbalance_ratio, 2),
                'minority_class': {
                    'name': str(value_counts.idxmin()),
                    'count': int(value_counts.min()),
                    'percentage': round(min_prop * 100, 2)
                },
                'majority_class': {
                    'name': str(value_counts.idxmax()),
                    'count': int(value_counts.max()),
                    'percentage': round(max_prop * 100, 2)
                },
                'class_details': class_info
            }
            
        except Exception as e:
            return {'error': f"Class balance analysis failed: {str(e)}"}
    
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target"""
        try:
            # Basic statistics
            stats_dict = {
                'count': len(target),
                'mean': round(target.mean(), 4),
                'median': round(target.median(), 4),
                'std': round(target.std(), 4),
                'min': round(target.min(), 4),
                'max': round(target.max(), 4),
                'range': round(target.max() - target.min(), 4),
                'iqr': round(target.quantile(0.75) - target.quantile(0.25), 4),
                'skewness': round(stats.skew(target), 4),
                'kurtosis': round(stats.kurtosis(target), 4)
            }
            
            # Distribution analysis
            distribution_analysis = self._analyze_target_distribution(target)
            
            # Outlier analysis
            outlier_analysis = self._robust_outlier_detection(target)
            
            return {
                'statistics': stats_dict,
                'distribution': distribution_analysis,
                'outliers': outlier_analysis,
                'recommendations': self._get_regression_recommendations(stats_dict, distribution_analysis)
            }
            
        except Exception as e:
            return {'error': f"Regression analysis failed: {str(e)}"}
    
    def _analyze_target_distribution(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze target distribution"""
        try:
            # Normality tests
            normality = self._test_normality(target)
            
            # Distribution shape
            skewness = stats.skew(target)
            kurtosis = stats.kurtosis(target)
            
            # Determine distribution type
            if abs(skewness) < 0.5:
                shape = 'approximately_normal'
            elif skewness > 1:
                shape = 'right_skewed'
            elif skewness < -1:
                shape = 'left_skewed'
            else:
                shape = 'moderately_skewed'
            
            return {
                'shape': shape,
                'normality': normality,
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'transformation_needed': abs(skewness) > 1 or not normality['is_normal']
            }
            
        except Exception as e:
            return {'error': f"Distribution analysis failed: {str(e)}"}
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality"""
        try:
            # Use appropriate test based on sample size
            if len(data) > 5000:
                # Use Kolmogorov-Smirnov for large samples
                stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                test_name = 'Kolmogorov-Smirnov'
            else:
                # Use Shapiro-Wilk for smaller samples
                stat, p_value = stats.shapiro(data[:5000])  # Limit to 5000 for performance
                test_name = 'Shapiro-Wilk'
            
            return {
                'is_normal': p_value > 0.05,
                'test_statistic': round(stat, 6),
                'p_value': round(p_value, 6),
                'test_name': test_name
            }
            
        except Exception as e:
            return {'error': f"Normality test failed: {str(e)}", 'is_normal': False}
    
    def _robust_outlier_detection(self, data: pd.Series) -> Dict[str, Any]:
        """Robust outlier detection using multiple methods"""
        try:
            outlier_info = {}
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_info['iqr_method'] = {
                'count': int(iqr_outliers),
                'percentage': round((iqr_outliers / len(data)) * 100, 2),
                'bounds': {'lower': round(lower_bound, 4), 'upper': round(upper_bound, 4)}
            }
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            outlier_info['zscore_method'] = {
                'count': int(z_outliers),
                'percentage': round((z_outliers / len(data)) * 100, 2),
                'threshold': 3
            }
            
            # Modified Z-score (using median)
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
            outlier_info['modified_zscore_method'] = {
                'count': int(modified_z_outliers),
                'percentage': round((modified_z_outliers / len(data)) * 100, 2),
                'threshold': 3.5
            }
            
            return outlier_info
            
        except Exception as e:
            return {'error': f"Outlier detection failed: {str(e)}"}
    
    def _find_robust_target_correlations(self, target: pd.Series) -> List[Dict[str, Any]]:
        """Find robust correlations with target"""
        correlations = []
        
        try:
            # Correlations with numeric features
            for col in self.numeric_cols:
                try:
                    feature_data = self.df[col].dropna()
                    if len(feature_data) > 10:  # Minimum data points
                        
                        # Align indices
                        common_idx = target.index.intersection(feature_data.index)
                        if len(common_idx) > 10:
                            target_aligned = target.loc[common_idx]
                            feature_aligned = feature_data.loc[common_idx]
                            
                            # Pearson correlation
                            try:
                                pearson_corr = target_aligned.corr(feature_aligned)
                                if not pd.isna(pearson_corr) and abs(pearson_corr) > 0.1:
                                    correlations.append({
                                        'feature': col,
                                        'correlation': round(pearson_corr, 4),
                                        'correlation_type': 'pearson',
                                        'abs_correlation': abs(pearson_corr)
                                    })
                            except:
                                pass
                            
                            # Spearman correlation (rank-based)
                            try:
                                spearman_corr = target_aligned.corr(feature_aligned, method='spearman')
                                if not pd.isna(spearman_corr) and abs(spearman_corr) > 0.1:
                                    correlations.append({
                                        'feature': col,
                                        'correlation': round(spearman_corr, 4),
                                        'correlation_type': 'spearman',
                                        'abs_correlation': abs(spearman_corr)
                                    })
                            except:
                                pass
                except:
                    continue
            
            # Associations with categorical features
            for col in self.categorical_cols:
                try:
                    assoc_strength = self._calculate_categorical_association(target, self.df[col])
                    if assoc_strength > 0.1:
                        correlations.append({
                            'feature': col,
                            'correlation': round(assoc_strength, 4),
                            'correlation_type': 'categorical_association',
                            'abs_correlation': assoc_strength
                        })
                except:
                    continue
            
            # Sort by absolute correlation and remove duplicates
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            # Remove duplicate features (keep highest correlation)
            seen_features = set()
            unique_correlations = []
            for corr in correlations:
                if corr['feature'] not in seen_features:
                    unique_correlations.append(corr)
                    seen_features.add(corr['feature'])
            
            return unique_correlations[:10]  # Top 10
            
        except Exception as e:
            return [{'error': f"Correlation analysis failed: {str(e)}"}]
    
    def _calculate_categorical_association(self, target: pd.Series, feature: pd.Series) -> float:
        """Calculate association between categorical variables"""
        try:
            # Create contingency table
            contingency = pd.crosstab(target, feature)
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V (normalized chi-square)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            return cramers_v
            
        except:
            return 0.0
    
    def _calculate_entropy(self, proportions: pd.Series) -> float:
        """Calculate entropy"""
        try:
            entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            return round(entropy, 4)
        except:
            return 0.0
    
    def _calculate_gini_impurity(self, proportions: pd.Series) -> float:
        """Calculate Gini impurity"""
        try:
            gini = 1 - sum(p**2 for p in proportions)
            return round(gini, 4)
        except:
            return 0.0
    
    def _get_classification_recommendations(self, balance_analysis: Dict) -> List[str]:
        """Get recommendations for classification tasks"""
        recommendations = []
        
        try:
            severity = balance_analysis.get('severity', 'low')
            status = balance_analysis.get('status', 'balanced')
            
            if status == 'single_class':
                recommendations.append("CRITICAL: Only one class present - classification impossible")
                recommendations.append("Consider collecting more diverse data or changing problem type")
                
            elif severity == 'high':
                recommendations.append("Severe class imbalance detected")
                recommendations.append("Apply SMOTE, ADASYN, or other resampling techniques")
                recommendations.append("Use stratified sampling for train/test splits")
                recommendations.append("Consider cost-sensitive learning algorithms")
                
            elif severity == 'medium':
                recommendations.append("Moderate class imbalance present")
                recommendations.append("Consider resampling or class weight adjustments")
                recommendations.append("Use appropriate evaluation metrics (F1-score, AUC-ROC)")
                
            else:
                recommendations.append("Classes are reasonably balanced")
                recommendations.append("Standard classification algorithms should work well")
                
        except:
            recommendations.append("Unable to generate specific recommendations")
            
        return recommendations
    
    def _get_regression_recommendations(self, stats: Dict, distribution: Dict) -> List[str]:
        """Get recommendations for regression tasks"""
        recommendations = []
        
        try:
            # Skewness recommendations
            skewness = abs(stats.get('skewness', 0))
            if skewness > 2:
                recommendations.append("Highly skewed target - Consider log transformation")
            elif skewness > 1:
                recommendations.append("Moderately skewed target - Consider sqrt or Box-Cox transformation")
            
            # Outlier recommendations
            if stats.get('std', 0) > stats.get('mean', 0) * 2:
                recommendations.append("High variance detected - Check for outliers")
                recommendations.append("Consider robust regression algorithms")
            
            # Distribution normality
            if not distribution.get('normality', {}).get('is_normal', True):
                recommendations.append("Target not normally distributed")
                recommendations.append("Consider non-parametric models or data transformation")
            
            if not recommendations:
                recommendations.append("Target distribution looks good for regression")
                
        except:
            recommendations.append("Unable to generate specific recommendations")
            
        return recommendations
    
    def _safe_analyze_feature_patterns(self):
        """Safely analyze feature patterns"""
        try:
            patterns = {}
            
            # Analyze numerical features
            if self.numeric_cols:
                patterns['numerical_patterns'] = self._analyze_numerical_patterns()
            
            # Analyze categorical features
            if self.categorical_cols:
                patterns['categorical_patterns'] = self._analyze_categorical_patterns()
            
            # Analyze datetime features
            if self.datetime_cols:
                patterns['datetime_patterns'] = self._analyze_datetime_patterns()
            
            # Analyze boolean features
            if self.boolean_cols:
                patterns['boolean_patterns'] = self._analyze_boolean_patterns()
            
            # Feature correlations
            patterns['feature_correlations'] = self._find_strong_correlations()
            
            # Feature importance (if target available)
            if self.target_column:
                patterns['feature_importance'] = self._calculate_feature_importance()
            
            self.insights['feature_patterns'] = patterns
            
        except Exception as e:
            self.insights['feature_patterns'] = {'error': f"Feature pattern analysis failed: {str(e)}"}
    
    def _analyze_numerical_patterns(self) -> Dict[str, Any]:
        """Analyze numerical feature patterns"""
        patterns = {}
        
        for col in self.numeric_cols:
            try:
                col_patterns = {}
                data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                
                if len(data) > 0:
                    # Basic statistics
                    col_patterns['count'] = len(data)
                    col_patterns['mean'] = round(data.mean(), 4)
                    col_patterns['std'] = round(data.std(), 4)
                    col_patterns['min'] = round(data.min(), 4)
                    col_patterns['max'] = round(data.max(), 4)
                    
                    # Distribution characteristics
                    col_patterns['skewness'] = round(stats.skew(data), 4)
                    col_patterns['kurtosis'] = round(stats.kurtosis(data), 4)
                    col_patterns['highly_skewed'] = abs(stats.skew(data)) > 2
                    col_patterns['moderately_skewed'] = 1 < abs(stats.skew(data)) <= 2
                    
                    # Value characteristics
                    col_patterns['zero_values'] = int((data == 0).sum())
                    col_patterns['zero_percentage'] = round(((data == 0).sum() / len(data)) * 100, 2)
                    col_patterns['negative_values'] = int((data < 0).sum())
                    col_patterns['negative_percentage'] = round(((data < 0).sum() / len(data)) * 100, 2)
                    
                    # Outlier analysis
                    col_patterns['outliers'] = self._robust_outlier_detection(data)
                    
                    # Variance analysis
                    col_patterns['variance'] = round(data.var(), 6)
                    col_patterns['coefficient_of_variation'] = round(data.std() / data.mean() if data.mean() != 0 else 0, 4)
                    col_patterns['low_variance'] = data.var() < 0.01
                    
                    # Unique values
                    col_patterns['unique_count'] = int(data.nunique())
                    col_patterns['unique_ratio'] = round(data.nunique() / len(data), 4)
                    
                    # Potential issues
                    issues = []
                    if col_patterns['highly_skewed']:
                        issues.append("highly_skewed")
                    if col_patterns['low_variance']:
                        issues.append("low_variance")
                    if col_patterns['outliers']['iqr_method']['percentage'] > 10:
                        issues.append("many_outliers")
                    if col_patterns['zero_percentage'] > 50:
                        issues.append("mostly_zeros")
                    
                    col_patterns['potential_issues'] = issues
                    
                patterns[col] = col_patterns
                
            except Exception as e:
                patterns[col] = {'error': f"Analysis failed: {str(e)}"}
        
        return patterns
    
    def _analyze_categorical_patterns(self) -> Dict[str, Any]:
        """Analyze categorical feature patterns"""
        patterns = {}
        
        for col in self.categorical_cols:
            try:
                col_patterns = {}
                data = self.df[col].dropna().astype(str)
                
                if len(data) > 0:
                    value_counts = data.value_counts()
                    
                    # Basic statistics
                    col_patterns['count'] = len(data)
                    col_patterns['unique_values'] = len(value_counts)
                    col_patterns['unique_ratio'] = round(len(value_counts) / len(data), 4)
                    
                    # Cardinality analysis
                    col_patterns['high_cardinality'] = len(value_counts) > len(data) * 0.5
                    col_patterns['very_high_cardinality'] = len(value_counts) > len(data) * 0.8
                    col_patterns['potential_id_column'] = len(value_counts) / len(data) > 0.95
                    
                    # Distribution analysis
                    if len(value_counts) > 0:
                        dominant_percentage = (value_counts.iloc[0] / len(data)) * 100
                        col_patterns['dominant_value'] = str(value_counts.index[0])
                        col_patterns['dominant_percentage'] = round(dominant_percentage, 2)
                        col_patterns['highly_dominant'] = dominant_percentage > 90
                        col_patterns['moderately_dominant'] = 70 <= dominant_percentage <= 90
                        
                        # Calculate entropy for distribution uniformity
                        proportions = value_counts / len(data)
                        col_patterns['entropy'] = self._calculate_entropy(proportions)
                        col_patterns['uniform_distribution'] = col_patterns['entropy'] > np.log2(len(value_counts)) * 0.9
                    
                    # Binary detection
                    col_patterns['is_binary'] = len(value_counts) == 2
                    if col_patterns['is_binary']:
                        col_patterns['binary_balance'] = round(min(value_counts) / max(value_counts), 3)
                    
                    # Common categorical patterns
                    col_patterns['has_null_like_values'] = any(val.lower() in ['null', 'none', 'na', 'n/a', 'unknown', 'missing', ''] for val in value_counts.index)
                    
                    # Value length analysis
                    value_lengths = data.str.len()
                    col_patterns['avg_value_length'] = round(value_lengths.mean(), 2)
                    col_patterns['max_value_length'] = int(value_lengths.max())
                    col_patterns['min_value_length'] = int(value_lengths.min())
                    
                    # Potential encoding issues
                    encoding_issues = []
                    if len(set(data.str.lower())) < len(set(data)):
                        encoding_issues.append("case_sensitivity")
                    if any(' ' in str(val) and str(val).strip() != str(val) for val in value_counts.index):
                        encoding_issues.append("whitespace_issues")
                    
                    col_patterns['encoding_issues'] = encoding_issues
                    
                    # Top categories
                    col_patterns['top_categories'] = dict(value_counts.head(10))
                    
                    # Potential issues
                    issues = []
                    if col_patterns['high_cardinality']:
                        issues.append("high_cardinality")
                    if col_patterns['highly_dominant']:
                        issues.append("single_value_dominant")
                    if col_patterns['potential_id_column']:
                        issues.append("likely_id_column")
                    if encoding_issues:
                        issues.extend(encoding_issues)
                    
                    col_patterns['potential_issues'] = issues
                    
                patterns[col] = col_patterns
                
            except Exception as e:
                patterns[col] = {'error': f"Analysis failed: {str(e)}"}
        
        return patterns
    
    def _analyze_datetime_patterns(self) -> Dict[str, Any]:
        """Analyze datetime feature patterns"""
        patterns = {}
        
        for col in self.datetime_cols:
            try:
                col_patterns = {}
                data = pd.to_datetime(self.df[col], errors='coerce').dropna()
                
                if len(data) > 0:
                    col_patterns['count'] = len(data)
                    col_patterns['min_date'] = str(data.min())
                    col_patterns['max_date'] = str(data.max())
                    col_patterns['date_range_days'] = (data.max() - data.min()).days
                    
                    # Frequency analysis
                    col_patterns['unique_dates'] = data.nunique()
                    col_patterns['duplicate_dates'] = len(data) - data.nunique()
                    
                    # Pattern analysis
                    col_patterns['has_time_component'] = any(data.dt.hour != 0) or any(data.dt.minute != 0)
                    col_patterns['spans_multiple_years'] = data.dt.year.nunique() > 1
                    col_patterns['business_hours_only'] = all((data.dt.hour >= 9) & (data.dt.hour <= 17))
                    
                patterns[col] = col_patterns
                
            except Exception as e:
                patterns[col] = {'error': f"DateTime analysis failed: {str(e)}"}
        
        return patterns
    
    def _analyze_boolean_patterns(self) -> Dict[str, Any]:
        """Analyze boolean feature patterns"""
        patterns = {}
        
        for col in self.boolean_cols:
            try:
                col_patterns = {}
                data = self.df[col].dropna()
                
                if len(data) > 0:
                    value_counts = data.value_counts()
                    
                    col_patterns['count'] = len(data)
                    col_patterns['true_count'] = int(value_counts.get(True, value_counts.iloc[0] if 'true' in str(value_counts.index[0]).lower() else 0))
                    col_patterns['false_count'] = int(len(data) - col_patterns['true_count'])
                    col_patterns['true_percentage'] = round((col_patterns['true_count'] / len(data)) * 100, 2)
                    col_patterns['balance_ratio'] = round(min(col_patterns['true_count'], col_patterns['false_count']) / max(col_patterns['true_count'], col_patterns['false_count']), 3)
                    col_patterns['well_balanced'] = col_patterns['balance_ratio'] > 0.3
                    
                patterns[col] = col_patterns
                
            except Exception as e:
                patterns[col] = {'error': f"Boolean analysis failed: {str(e)}"}
        
        return patterns
    
    def _find_strong_correlations(self) -> List[Dict[str, Any]]:
        """Find strong correlations between features"""
        correlations = []
        
        try:
            if len(self.numeric_cols) > 1:
                # Calculate correlation matrix
                numeric_data = self.df[self.numeric_cols]
                corr_matrix = numeric_data.corr()
                
                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.7:
                            correlations.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': round(corr_val, 4),
                                'correlation_strength': 'very_strong' if abs(corr_val) > 0.9 else 'strong',
                                'correlation_type': 'positive' if corr_val > 0 else 'negative'
                            })
                
                # Sort by absolute correlation
                correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
        except Exception as e:
            correlations = [{'error': f"Correlation analysis failed: {str(e)}"}]
        
        return correlations[:20]  # Top 20 correlations
    
    def _calculate_feature_importance(self) -> Dict[str, Any]:
        """Calculate feature importance relative to target"""
        importance_scores = {}
        
        try:
            if not self.target_column or self.target_column not in self.df.columns:
                return {'error': 'No valid target column for importance calculation'}
            
            target = self.df[self.target_column].dropna()
            
            # For numerical features
            numeric_importance = []
            for col in self.numeric_cols:
                try:
                    feature_data = pd.to_numeric(self.df[col], errors='coerce')
                    common_idx = target.index.intersection(feature_data.dropna().index)
                    
                    if len(common_idx) > 10:
                        target_aligned = target.loc[common_idx]
                        feature_aligned = feature_data.loc[common_idx]
                        
                        # Use correlation as importance measure
                        importance = abs(target_aligned.corr(feature_aligned))
                        if not pd.isna(importance):
                            numeric_importance.append({
                                'feature': col,
                                'importance': round(importance, 4),
                                'type': 'correlation'
                            })
                except:
                    continue
            
            # For categorical features
            categorical_importance = []
            for col in self.categorical_cols:
                try:
                    assoc_strength = self._calculate_categorical_association(target, self.df[col])
                    categorical_importance.append({
                        'feature': col,
                        'importance': round(assoc_strength, 4),
                        'type': 'association'
                    })
                except:
                    continue
            
            # Combine and sort
            all_importance = numeric_importance + categorical_importance
            all_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            importance_scores = {
                'numerical_features': numeric_importance,
                'categorical_features': categorical_importance,
                'top_features': all_importance[:10],
                'method': 'correlation_and_association'
            }
            
        except Exception as e:
            importance_scores = {'error': f"Feature importance calculation failed: {str(e)}"}
        
        return importance_scores
    
    def _safe_generate_statistical_warnings(self):
        """Generate statistical warnings safely"""
        try:
            warnings = []
            
            # Check each column for various issues
            for col in self.df.columns:
                warnings.extend(self._check_column_warnings(col))
            
            # Add general dataset warnings
            warnings.extend(self._check_dataset_warnings())
            
            # Sort by severity
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            warnings.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 0), reverse=True)
            
            self.insights['statistical_warnings'] = warnings
            
        except Exception as e:
            self.insights['statistical_warnings'] = [{'error': f"Warning generation failed: {str(e)}"}]
    
    def _check_column_warnings(self, col: str) -> List[Dict[str, Any]]:
        """Check warnings for a specific column"""
        warnings = []
        
        try:
            data = self.df[col]
            missing_pct = (data.isnull().sum() / len(data)) * 100
            
            # Missing value warnings
            if missing_pct > 90:
                warnings.append({
                    'type': 'critical_missing',
                    'column': col,
                    'message': f"Column '{col}' is {missing_pct:.1f}% missing - Consider dropping",
                    'severity': 'high',
                    'value': missing_pct
                })
            elif missing_pct > 50:
                warnings.append({
                    'type': 'high_missing',
                    'column': col,
                    'message': f"Column '{col}' is {missing_pct:.1f}% missing - Requires attention",
                    'severity': 'medium',
                    'value': missing_pct
                })
            
            # Column-specific warnings based on type
            if col in self.numeric_cols:
                warnings.extend(self._check_numeric_column_warnings(col, data))
            elif col in self.categorical_cols:
                warnings.extend(self._check_categorical_column_warnings(col, data))
            
        except Exception as e:
            warnings.append({
                'type': 'analysis_error',
                'column': col,
                'message': f"Could not analyze column '{col}': {str(e)}",
                'severity': 'low'
            })
        
        return warnings
    
    def _check_numeric_column_warnings(self, col: str, data: pd.Series) -> List[Dict[str, Any]]:
        """Check warnings specific to numeric columns"""
        warnings = []
        
        try:
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            
            if len(numeric_data) == 0:
                return warnings
                
            # Variance warnings
            if numeric_data.var() < 1e-10:
                warnings.append({
                    'type': 'zero_variance',
                    'column': col,
                    'message': f"Column '{col}' has zero or near-zero variance - Remove for ML",
                    'severity': 'medium',
                    'value': numeric_data.var()
                })
            
            # Outlier warnings
            outlier_info = self._robust_outlier_detection(numeric_data)
            iqr_outlier_pct = outlier_info.get('iqr_method', {}).get('percentage', 0)
            
            if iqr_outlier_pct > 20:
                warnings.append({
                    'type': 'many_outliers',
                    'column': col,
                    'message': f"Column '{col}' has {iqr_outlier_pct:.1f}% outliers - Consider capping or transformation",
                    'severity': 'medium',
                    'value': iqr_outlier_pct
                })
            
            # Skewness warnings
            skewness = abs(stats.skew(numeric_data))
            if skewness > 3:
                warnings.append({
                    'type': 'extreme_skewness',
                    'column': col,
                    'message': f"Column '{col}' is extremely skewed (skew={skewness:.2f}) - Consider log transformation",
                    'severity': 'medium',
                    'value': skewness
                })
            
            # Suspicious values
            suspicious_count = numeric_data.isin([-999, -99, 999999, -1]).sum()
            if suspicious_count > 0:
                warnings.append({
                    'type': 'suspicious_values',
                    'column': col,
                    'message': f"Column '{col}' has {suspicious_count} potential placeholder values (-999, -99, etc.)",
                    'severity': 'high',
                    'value': suspicious_count
                })
            
        except Exception as e:
            warnings.append({
                'type': 'numeric_analysis_error',
                'column': col,
                'message': f"Numeric analysis failed for '{col}': {str(e)}",
                'severity': 'low'
            })
        
        return warnings
    
    def _check_categorical_column_warnings(self, col: str, data: pd.Series) -> List[Dict[str, Any]]:
        """Check warnings specific to categorical columns"""
        warnings = []
        
        try:
            clean_data = data.dropna().astype(str)
            
            if len(clean_data) == 0:
                return warnings
            
            value_counts = clean_data.value_counts()
            
            # High cardinality warning
            cardinality_ratio = len(value_counts) / len(clean_data)
            if cardinality_ratio > 0.8:
                warnings.append({
                    'type': 'high_cardinality',
                    'column': col,
                    'message': f"Column '{col}' has very high cardinality ({len(value_counts)} unique values) - Consider encoding or grouping",
                    'severity': 'medium',
                    'value': len(value_counts)
                })
            
            # Single value dominance
            if len(value_counts) > 0:
                dominant_pct = (value_counts.iloc[0] / len(clean_data)) * 100
                if dominant_pct > 95:
                    warnings.append({
                        'type': 'single_value_dominance',
                        'column': col,
                        'message': f"Column '{col}' is {dominant_pct:.1f}% single value ('{value_counts.index[0]}') - May be uninformative",
                        'severity': 'medium',
                        'value': dominant_pct
                    })
            
            # Potential ID column
            if cardinality_ratio > 0.95:
                warnings.append({
                    'type': 'potential_id_column',
                    'column': col,
                    'message': f"Column '{col}' appears to be an ID column - Remove for ML analysis",
                    'severity': 'low',
                    'value': cardinality_ratio
                })
            
        except Exception as e:
            warnings.append({
                'type': 'categorical_analysis_error',
                'column': col,
                'message': f"Categorical analysis failed for '{col}': {str(e)}",
                'severity': 'low'
            })
        
        return warnings
    
    def _check_dataset_warnings(self) -> List[Dict[str, Any]]:
        """Check general dataset warnings"""
        warnings = []
        
        try:
            # Small dataset warning
            if self.df.shape[0] < 100:
                warnings.append({
                    'type': 'small_dataset',
                    'column': 'dataset',
                    'message': f"Dataset has only {self.df.shape[0]} rows - May be too small for reliable ML",
                    'severity': 'medium',
                    'value': self.df.shape[0]
                })
            
            # Too many features warning
            if self.df.shape[1] > self.df.shape[0]:
                warnings.append({
                    'type': 'high_dimensionality',
                    'column': 'dataset',
                    'message': f"More features ({self.df.shape[1]}) than samples ({self.df.shape[0]}) - Curse of dimensionality",
                    'severity': 'high',
                    'value': self.df.shape[1] / self.df.shape[0]
                })
            
        except Exception as e:
            warnings.append({
                'type': 'dataset_analysis_error',
                'column': 'dataset',
                'message': f"Dataset analysis failed: {str(e)}",
                'severity': 'low'
            })
        
        return warnings
    
    def _safe_analyze_data_quality(self):
        """Safely analyze data quality"""
        try:
            quality_insights = []
            
            # Check for data quality issues across different column types
            quality_insights.extend(self._check_data_consistency())
            quality_insights.extend(self._check_data_validity())
            quality_insights.extend(self._check_encoding_issues())
            
            self.insights['data_quality'] = quality_insights
            
        except Exception as e:
            self.insights['data_quality'] = [f"Data quality analysis failed: {str(e)}"]
    
    def _check_data_consistency(self) -> List[str]:
        """Check for data consistency issues"""
        issues = []
        
        try:
            # Check for mixed data types in categorical columns
            for col in self.categorical_cols:
                try:
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        # Check for mixed numeric and text
                        numeric_count = pd.to_numeric(data, errors='coerce').notna().sum()
                        if 0 < numeric_count < len(data):
                            issues.append(f"Column '{col}' contains mixed numeric and text values - {numeric_count}/{len(data)} are numeric")
                except:
                    continue
            
            # Check for inconsistent date formats
            for col in self.datetime_cols:
                try:
                    original_data = self.df[col].dropna().astype(str)
                    if len(original_data) > 0:
                        # Check for different date patterns
                        date_patterns = set()
                        for date_str in original_data.head(100):  # Sample first 100
                            if '/' in date_str:
                                date_patterns.add('slash_format')
                            elif '-' in date_str:
                                date_patterns.add('dash_format')
                            elif '.' in date_str:
                                date_patterns.add('dot_format')
                        
                        if len(date_patterns) > 1:
                            issues.append(f"Column '{col}' has inconsistent date formats: {', '.join(date_patterns)}")
                except:
                    continue
            
        except Exception as e:
            issues.append(f"Consistency check failed: {str(e)}")
        
        return issues
    
    def _check_data_validity(self) -> List[str]:
        """Check for data validity issues"""
        issues = []
        
        try:
            # Check for unrealistic values in common column types
            for col in self.df.columns:
                col_lower = col.lower()
                
                # Age columns
                if 'age' in col_lower and col in self.numeric_cols:
                    try:
                        age_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                        invalid_ages = ((age_data < 0) | (age_data > 150)).sum()
                        if invalid_ages > 0:
                            issues.append(f"Column '{col}' has {invalid_ages} unrealistic age values (< 0 or > 150)")
                    except:
                        continue
                
                # Price/Amount columns
                elif any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'salary', 'income', 'revenue']) and col in self.numeric_cols:
                    try:
                        money_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                        negative_values = (money_data < 0).sum()
                        if negative_values > 0:
                            issues.append(f"Column '{col}' has {negative_values} negative values which may be unrealistic for monetary data")
                    except:
                        continue
                
                # Percentage columns
                elif any(keyword in col_lower for keyword in ['percent', 'rate', 'ratio']) and col in self.numeric_cols:
                    try:
                        pct_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
                        invalid_pct = ((pct_data < 0) | (pct_data > 100)).sum()
                        if invalid_pct > 0:
                            issues.append(f"Column '{col}' has {invalid_pct} values outside 0-100% range")
                    except:
                        continue
            
        except Exception as e:
            issues.append(f"Validity check failed: {str(e)}")
        
        return issues
    
    def _check_encoding_issues(self) -> List[str]:
        """Check for encoding and formatting issues"""
        issues = []
        
        try:
            for col in self.categorical_cols:
                try:
                    data = self.df[col].dropna().astype(str)
                    if len(data) > 0:
                        # Check for leading/trailing spaces
                        space_issues = (data != data.str.strip()).sum()
                        if space_issues > 0:
                            issues.append(f"Column '{col}' has {space_issues} values with leading/trailing spaces")
                        
                        # Check for case inconsistencies
                        unique_original = data.nunique()
                        unique_lower = data.str.lower().nunique()
                        if unique_lower < unique_original:
                            issues.append(f"Column '{col}' has case sensitivity issues - {unique_original} vs {unique_lower} unique values")
                        
                        # Check for special characters that might indicate encoding issues
                        special_char_pattern = r'[^\w\s\-\.\,\(\)\[\]]'
                        special_chars = data.str.contains(special_char_pattern, regex=True, na=False).sum()
                        if special_chars > len(data) * 0.1:  # More than 10% have special chars
                            issues.append(f"Column '{col}' may have encoding issues - {special_chars} values contain unusual characters")
                        
                        # Check for unicode issues
                        unicode_issues = data.str.contains(r'\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}', regex=True, na=False).sum()
                        if unicode_issues > 0:
                            issues.append(f"Column '{col}' has {unicode_issues} values with unicode escape sequences")
                        
                        # Check for null-like strings
                        null_like_values = ['null', 'none', 'na', 'n/a', 'nan', 'nil', 'undefined', '']
                        null_like_count = data.str.lower().isin(null_like_values).sum()
                        if null_like_count > 0:
                            issues.append(f"Column '{col}' has {null_like_count} null-like string values that should be actual nulls")
                        
                        # Check for inconsistent separators in potential list-like data
                        if data.str.contains('[,;|]', regex=True, na=False).sum() > len(data) * 0.3:
                            separator_types = []
                            if data.str.contains(',', na=False).any():
                                separator_types.append('comma')
                            if data.str.contains(';', na=False).any():
                                separator_types.append('semicolon')
                            if data.str.contains('|', na=False).any():
                                separator_types.append('pipe')
                            
                            if len(separator_types) > 1:
                                issues.append(f"Column '{col}' has inconsistent separators: {', '.join(separator_types)}")
                            
                except Exception as e:
                    issues.append(f"Encoding check failed for column '{col}': {str(e)}")
                    continue
            
        except Exception as e:
            issues.append(f"Overall encoding check failed: {str(e)}")
        
        return issues
    
    def _safe_discover_relationships(self):
        """Safely discover relationships between features"""
        try:
            relationships = {
                'strong_correlations': [],
                'feature_interactions': [],
                'dependency_patterns': [],
                'clustering_insights': [],
                'multicollinearity_warnings': []
            }
            
            # Discover numeric relationships
            if len(self.numeric_cols) > 1:
                relationships['strong_correlations'] = self._find_detailed_correlations()
                relationships['multicollinearity_warnings'] = self._detect_multicollinearity()
            
            # Discover categorical relationships
            if len(self.categorical_cols) > 1:
                relationships['dependency_patterns'] = self._find_categorical_dependencies()
            
            # Cross-type relationships
            if self.numeric_cols and self.categorical_cols:
                relationships['feature_interactions'] = self._find_mixed_type_interactions()
            
            # Advanced pattern discovery
            relationships['clustering_insights'] = self._discover_clustering_patterns()
            
            self.insights['relationships'] = relationships
            
        except Exception as e:
            self.insights['relationships'] = {'error': f"Relationship discovery failed: {str(e)}"}
    
    def _find_detailed_correlations(self) -> List[Dict[str, Any]]:
        """Find detailed correlations with statistical significance"""
        correlations = []
        
        try:
            numeric_data = self.df[self.numeric_cols].select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return correlations
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Find significant correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if not pd.isna(corr_val) and abs(corr_val) > 0.3:
                        # Calculate statistical significance
                        try:
                            common_data = numeric_data[[col1, col2]].dropna()
                            if len(common_data) > 10:
                                # Pearson correlation with p-value
                                pearson_corr, p_value = stats.pearsonr(common_data[col1], common_data[col2])
                                
                                # Spearman correlation for non-linear relationships
                                spearman_corr, _ = stats.spearmanr(common_data[col1], common_data[col2])
                                
                                correlation_info = {
                                    'feature1': col1,
                                    'feature2': col2,
                                    'pearson_correlation': round(pearson_corr, 4),
                                    'spearman_correlation': round(spearman_corr, 4),
                                    'p_value': round(p_value, 6),
                                    'is_significant': p_value < 0.05,
                                    'sample_size': len(common_data),
                                    'strength': self._classify_correlation_strength(abs(pearson_corr)),
                                    'relationship_type': 'positive' if pearson_corr > 0 else 'negative'
                                }
                                
                                correlations.append(correlation_info)
                        except:
                            continue
            
            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x['pearson_correlation']), reverse=True)
            return correlations[:15]  # Top 15 correlations
            
        except Exception as e:
            return [{'error': f"Detailed correlation analysis failed: {str(e)}"}]
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength"""
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _detect_multicollinearity(self) -> List[Dict[str, Any]]:
        """Detect multicollinearity issues"""
        warnings = []
        
        try:
            numeric_data = self.df[self.numeric_cols].select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 3:
                return warnings
            
            # Calculate VIF (Variance Inflation Factor) approximation
            corr_matrix = numeric_data.corr()
            
            # Find groups of highly correlated features
            high_corr_groups = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if not pd.isna(corr_val) and corr_val > 0.8:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        
                        # Check if either feature is already in a group
                        added_to_group = False
                        for group in high_corr_groups:
                            if col1 in group or col2 in group:
                                group.update([col1, col2])
                                added_to_group = True
                                break
                        
                        if not added_to_group:
                            high_corr_groups.append({col1, col2})
            
            # Create warnings for each group
            for i, group in enumerate(high_corr_groups):
                if len(group) > 1:
                    warnings.append({
                        'type': 'multicollinearity_group',
                        'features': list(group),
                        'group_id': i + 1,
                        'severity': 'high' if len(group) > 2 else 'medium',
                        'recommendation': f"Consider removing redundant features from group: {', '.join(list(group))}"
                    })
            
        except Exception as e:
            warnings.append({'error': f"Multicollinearity detection failed: {str(e)}"})
        
        return warnings
    
    def _find_categorical_dependencies(self) -> List[Dict[str, Any]]:
        """Find dependencies between categorical variables"""
        dependencies = []
        
        try:
            # Test pairwise dependencies using Chi-square test
            for i in range(len(self.categorical_cols)):
                for j in range(i+1, len(self.categorical_cols)):
                    col1, col2 = self.categorical_cols[i], self.categorical_cols[j]
                    
                    try:
                        # Create contingency table
                        contingency = pd.crosstab(self.df[col1].fillna('Missing'), 
                                                self.df[col2].fillna('Missing'))
                        
                        # Skip if contingency table is too small
                        if contingency.size < 4:
                            continue
                        
                        # Chi-square test
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        
                        # Cramér's V for effect size
                        n = contingency.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                        
                        if p_value < 0.05 and cramers_v > 0.1:
                            dependencies.append({
                                'feature1': col1,
                                'feature2': col2,
                                'chi_square_statistic': round(chi2, 4),
                                'p_value': round(p_value, 6),
                                'cramers_v': round(cramers_v, 4),
                                'degrees_of_freedom': dof,
                                'is_significant': True,
                                'association_strength': self._classify_association_strength(cramers_v),
                                'interpretation': f"'{col1}' and '{col2}' show statistical dependency"
                            })
                    except:
                        continue
            
            # Sort by association strength
            dependencies.sort(key=lambda x: x['cramers_v'], reverse=True)
            return dependencies[:10]  # Top 10 dependencies
            
        except Exception as e:
            return [{'error': f"Categorical dependency analysis failed: {str(e)}"}]
    
    def _classify_association_strength(self, cramers_v: float) -> str:
        """Classify association strength based on Cramér's V"""
        if cramers_v >= 0.5:
            return 'very_strong'
        elif cramers_v >= 0.3:
            return 'strong'
        elif cramers_v >= 0.15:
            return 'moderate'
        elif cramers_v >= 0.05:
            return 'weak'
        else:
            return 'very_weak'
    
    def _find_mixed_type_interactions(self) -> List[Dict[str, Any]]:
        """Find interactions between different feature types"""
        interactions = []
        
        try:
            # Analyze categorical vs numeric relationships
            for cat_col in self.categorical_cols:
                for num_col in self.numeric_cols:
                    try:
                        interaction = self._analyze_categorical_numeric_interaction(cat_col, num_col)
                        if interaction and interaction.get('is_significant', False):
                            interactions.append(interaction)
                    except:
                        continue
            
            # Sort by effect size
            interactions.sort(key=lambda x: x.get('effect_size', 0), reverse=True)
            return interactions[:10]  # Top 10 interactions
            
        except Exception as e:
            return [{'error': f"Mixed-type interaction analysis failed: {str(e)}"}]
    
    def _analyze_categorical_numeric_interaction(self, cat_col: str, num_col: str) -> Dict[str, Any]:
        """Analyze interaction between categorical and numeric features"""
        try:
            # Prepare data
            cat_data = self.df[cat_col].fillna('Missing')
            num_data = pd.to_numeric(self.df[num_col], errors='coerce')
            
            # Remove rows where numeric data is missing
            valid_mask = num_data.notna()
            cat_data = cat_data[valid_mask]
            num_data = num_data[valid_mask]
            
            if len(num_data) < 10:
                return None
            
            # Group numeric data by categorical values
            groups = [num_data[cat_data == cat] for cat in cat_data.unique() if len(num_data[cat_data == cat]) > 2]
            
            if len(groups) < 2:
                return None
            
            # Perform ANOVA or Kruskal-Wallis test
            try:
                # Check normality assumption (simplified)
                normality_ok = all(len(group) < 30 or abs(stats.skew(group)) < 2 for group in groups)
                
                if normality_ok:
                    # Use ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    test_type = 'anova'
                else:
                    # Use Kruskal-Wallis (non-parametric)
                    h_stat, p_value = stats.kruskal(*groups)
                    f_stat = h_stat
                    test_type = 'kruskal_wallis'
                
                # Calculate effect size (eta-squared approximation)
                group_means = [group.mean() for group in groups]
                overall_mean = num_data.mean()
                
                ss_between = sum(len(group) * (mean - overall_mean)**2 for group, mean in zip(groups, group_means))
                ss_total = sum((num_data - overall_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                return {
                    'categorical_feature': cat_col,
                    'numeric_feature': num_col,
                    'test_type': test_type,
                    'test_statistic': round(f_stat, 4),
                    'p_value': round(p_value, 6),
                    'is_significant': p_value < 0.05,
                    'effect_size': round(eta_squared, 4),
                    'effect_interpretation': self._interpret_effect_size(eta_squared),
                    'group_count': len(groups),
                    'interpretation': f"'{cat_col}' has {'significant' if p_value < 0.05 else 'no significant'} effect on '{num_col}'"
                }
                
            except:
                return None
                
        except Exception as e:
            return {'error': f"Categorical-numeric interaction analysis failed: {str(e)}"}
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret effect size (eta-squared)"""
        if eta_squared >= 0.14:
            return 'large'
        elif eta_squared >= 0.06:
            return 'medium'
        elif eta_squared >= 0.01:
            return 'small'
        else:
            return 'negligible'
    
    def _discover_clustering_patterns(self) -> List[Dict[str, Any]]:
        """Discover clustering patterns in the data"""
        patterns = []
        
        try:
            # Only proceed if we have enough numeric features
            if len(self.numeric_cols) < 2:
                return [{'message': 'Not enough numeric features for clustering analysis'}]
            
            numeric_data = self.df[self.numeric_cols].select_dtypes(include=[np.number])
            
            # Remove rows with too many missing values
            numeric_data = numeric_data.dropna(thresh=len(numeric_data.columns) * 0.7)
            
            if len(numeric_data) < 10:
                return [{'message': 'Insufficient data for clustering analysis'}]
            
            # Simple clustering insights using correlation structure
            corr_matrix = numeric_data.corr()
            
            # Find potential feature clusters based on high correlations
            feature_clusters = self._find_feature_clusters(corr_matrix)
            
            for i, cluster in enumerate(feature_clusters):
                if len(cluster) > 1:
                    patterns.append({
                        'cluster_id': i + 1,
                        'features': list(cluster),
                        'cluster_size': len(cluster),
                        'pattern_type': 'feature_similarity',
                        'description': f"Features {', '.join(list(cluster))} show similar patterns"
                    })
            
            # Data distribution insights
            if len(numeric_data) > 0:
                # Check for potential outlier clusters
                outlier_insights = self._analyze_outlier_patterns(numeric_data)
                patterns.extend(outlier_insights)
            
        except Exception as e:
            patterns.append({'error': f"Clustering pattern discovery failed: {str(e)}"})
        
        return patterns
    
    def _find_feature_clusters(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[set]:
        """Find clusters of similar features based on correlation"""
        clusters = []
        processed_features = set()
        
        try:
            for i, feature1 in enumerate(corr_matrix.columns):
                if feature1 in processed_features:
                    continue
                
                cluster = {feature1}
                
                for j, feature2 in enumerate(corr_matrix.columns):
                    if i != j and feature2 not in processed_features:
                        if abs(corr_matrix.loc[feature1, feature2]) > threshold:
                            cluster.add(feature2)
                
                if len(cluster) > 1:
                    clusters.append(cluster)
                    processed_features.update(cluster)
                else:
                    processed_features.add(feature1)
        
        except Exception as e:
            pass
        
        return clusters
    
    def _analyze_outlier_patterns(self, numeric_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns in outliers"""
        patterns = []
        
        try:
            # Simple outlier detection using IQR method
            outlier_mask = pd.Series([False] * len(numeric_data), index=numeric_data.index)
            
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
            
            outlier_percentage = (outlier_mask.sum() / len(numeric_data)) * 100
            
            if outlier_percentage > 5:
                patterns.append({
                    'pattern_type': 'outlier_concentration',
                    'outlier_percentage': round(outlier_percentage, 2),
                    'outlier_count': int(outlier_mask.sum()),
                    'description': f"{outlier_percentage:.1f}% of data points are outliers",
                    'severity': 'high' if outlier_percentage > 15 else 'medium'
                })
            
        except Exception as e:
            patterns.append({'error': f"Outlier pattern analysis failed: {str(e)}"})
        
        return patterns
    
    def _safe_generate_ai_recommendations(self):
        """Generate AI-powered recommendations safely"""
        try:
            recommendations = {
                'data_preprocessing': [],
                'feature_engineering': [],
                'modeling_suggestions': [],
                'data_collection': [],
                'general_insights': []
            }
            
            # Data preprocessing recommendations
            recommendations['data_preprocessing'] = self._generate_preprocessing_recommendations()
            
            # Feature engineering recommendations
            recommendations['feature_engineering'] = self._generate_feature_engineering_recommendations()
            
            # Modeling recommendations
            recommendations['modeling_suggestions'] = self._generate_modeling_recommendations()
            
            # Data collection recommendations
            recommendations['data_collection'] = self._generate_data_collection_recommendations()
            
            # General insights
            recommendations['general_insights'] = self._generate_general_insights()
            
            self.insights['ai_recommendations'] = recommendations
            
        except Exception as e:
            self.insights['ai_recommendations'] = {'error': f"AI recommendation generation failed: {str(e)}"}
    
    def _generate_preprocessing_recommendations(self) -> List[Dict[str, Any]]:
        """Generate data preprocessing recommendations"""
        recommendations = []
        
        try:
            # Missing value recommendations
            missing_info = self.insights.get('data_composition', {}).get('missing_data', {})
            high_missing_cols = missing_info.get('high_missing_columns', [])
            
            if high_missing_cols:
                recommendations.append({
                    'category': 'missing_values',
                    'priority': 'high',
                    'action': 'Handle high missing value columns',
                    'details': f"Columns {high_missing_cols} have >50% missing values",
                    'specific_steps': [
                        "Consider dropping columns with >80% missing values",
                        "Use advanced imputation techniques (KNN, MICE) for important features",
                        "Create 'missing indicator' features for informative missingness"
                    ]
                })
            
            # Encoding recommendations
            if len(self.categorical_cols) > 0:
                high_cardinality_cols = []
                categorical_patterns = self.insights.get('feature_patterns', {}).get('categorical_patterns', {})
                
                for col, pattern in categorical_patterns.items():
                    if isinstance(pattern, dict) and pattern.get('high_cardinality', False):
                        high_cardinality_cols.append(col)
                
                if high_cardinality_cols:
                    recommendations.append({
                        'category': 'encoding',
                        'priority': 'medium',
                        'action': 'Address high cardinality categorical features',
                        'details': f"Columns {high_cardinality_cols} have high cardinality",
                        'specific_steps': [
                            "Use target encoding for high-cardinality categories",
                            "Group rare categories into 'Other' category",
                            "Consider embedding techniques for very high cardinality"
                        ]
                    })
            
            # Scaling recommendations
            if len(self.numeric_cols) > 1:
                recommendations.append({
                    'category': 'scaling',
                    'priority': 'medium',
                    'action': 'Scale numerical features',
                    'details': "Different numerical features may have different scales",
                    'specific_steps': [
                        "Use StandardScaler for normally distributed features",
                        "Use RobustScaler for features with outliers",
                        "Use MinMaxScaler for bounded distributions"
                    ]
                })
            
            # Outlier handling recommendations
            statistical_warnings = self.insights.get('statistical_warnings', [])
            outlier_warnings = [w for w in statistical_warnings if isinstance(w,dict) and w.get('type') == 'many_outliers']
            
            if outlier_warnings:
                recommendations.append({
                    'category': 'outliers',
                    'priority': 'medium',
                    'action': 'Handle outliers in numerical features',
                    'details': f"Found outlier warnings in {len(outlier_warnings)} columns",
                    'specific_steps': [
                        "Use IQR method for outlier detection",
                        "Consider winsorization (capping) instead of removal",
                        "Use robust algorithms that handle outliers naturally"
                    ]
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'error',
                'priority': 'low',
                'action': 'Preprocessing analysis incomplete',
                'details': f"Error in analysis: {str(e)}"
            })
        
        return recommendations
    
    def _generate_feature_engineering_recommendations(self) -> List[Dict[str, Any]]:
        """Generate feature engineering recommendations"""
        recommendations = []
        
        try:
            # Transformation recommendations
            numeric_patterns = self.insights.get('feature_patterns', {}).get('numerical_patterns', {})
            
            highly_skewed_cols = []
            for col, pattern in numeric_patterns.items():
                if isinstance(pattern, dict) and pattern.get('highly_skewed', False):
                    highly_skewed_cols.append(col)
            
            if highly_skewed_cols:
                recommendations.append({
                    'category': 'transformations',
                    'priority': 'medium',
                    'action': 'Transform highly skewed features',
                    'details': f"Columns {highly_skewed_cols} are highly skewed",
                    'specific_steps': [
                        "Apply log transformation for right-skewed data",
                        "Use Box-Cox transformation for optimal normalization",
                        "Consider square root transformation for moderate skewness"
                    ]
                })
            
            # Feature interaction recommendations
            strong_correlations = self.insights.get('relationships', {}).get('strong_correlations', [])
            
            if len(strong_correlations) > 0:
                recommendations.append({
                    'category': 'feature_interactions',
                    'priority': 'low',
                    'action': 'Create interaction features',
                    'details': f"Found {len(strong_correlations)} strong feature correlations",
                    'specific_steps': [
                        "Create polynomial features for strongly correlated pairs",
                        "Generate ratio features between related variables",
                        "Use feature crosses for categorical interactions"
                    ]
                })
            
            # Dimensionality recommendations
            if len(self.numeric_cols) > 10:
                recommendations.append({
                    'category': 'dimensionality',
                    'priority': 'medium',
                    'action': 'Consider dimensionality reduction',
                    'details': f"Dataset has {len(self.numeric_cols)} numerical features",
                    'specific_steps': [
                        "Apply PCA for linear dimensionality reduction",
                        "Use feature selection techniques (RFE, SelectKBest)",
                        "Consider t-SNE or UMAP for visualization"
                    ]
                })
            
            # Datetime feature engineering
            if len(self.datetime_cols) > 0:
                recommendations.append({
                    'category': 'datetime_features',
                    'priority': 'medium',
                    'action': 'Extract datetime features',
                    'details': f"Found {len(self.datetime_cols)} datetime columns",
                    'specific_steps': [
                        "Extract year, month, day, hour components",
                        "Create cyclical features (sin/cos) for periodic patterns",
                        "Generate time-since features for recency effects"
                    ]
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'error',
                'priority': 'low',
                'action': 'Feature engineering analysis incomplete',
                'details': f"Error in analysis: {str(e)}"
            })
        
        return recommendations
    
    def _generate_modeling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate modeling recommendations"""
        recommendations = []
        
        try:
            # Task-specific recommendations
            target_insights = self.insights.get('target_insights', {})
            task_type = target_insights.get('task_type')
            
            if task_type == 'classification':
                class_balance = target_insights.get('class_balance', {})
                
                if class_balance.get('severity') == 'high':
                    recommendations.append({
                        'category': 'classification_imbalanced',
                        'priority': 'high',
                        'action': 'Address severe class imbalance',
                        'details': f"Imbalance ratio: {class_balance.get('imbalance_ratio', 'N/A')}",
                        'specific_steps': [
                            "Use ensemble methods (RandomForest, XGBoost) with class weights",
                            "Apply SMOTE or ADASYN for synthetic sample generation",
                            "Use stratified sampling for train-test splits",
                            "Focus on precision-recall metrics over accuracy"
                        ]
                    })
                else:
                    recommendations.append({
                        'category': 'classification_balanced',
                        'priority': 'medium',
                        'action': 'Standard classification approaches',
                        'details': "Classes are reasonably balanced",
                        'specific_steps': [
                            "Try Logistic Regression as baseline",
                            "Use Random Forest for feature importance",
                            "Consider SVM for high-dimensional data",
                            "Use cross-validation for model selection"
                        ]
                    })
            
            elif task_type == 'regression':
                distribution = target_insights.get('distribution', {})
                
                if not distribution.get('normality', {}).get('is_normal', True):
                    recommendations.append({
                        'category': 'regression_non_normal',
                        'priority': 'medium',
                        'action': 'Handle non-normal target distribution',
                        'details': "Target variable is not normally distributed",
                        'specific_steps': [
                            "Consider tree-based models (RandomForest, XGBoost)",
                            "Use robust regression techniques",
                            "Transform target variable if needed",
                            "Evaluate using appropriate metrics (MAE vs RMSE)"
                        ]
                    })
                else:
                    recommendations.append({
                        'category': 'regression_normal',
                        'priority': 'medium',
                        'action': 'Standard regression approaches',
                        'details': "Target distribution is approximately normal",
                        'specific_steps': [
                            "Start with Linear Regression as baseline",
                            "Try Ridge/Lasso for regularization",
                            "Use ensemble methods for better performance",
                            "Consider polynomial features for non-linearity"
                        ]
                    })
            
            # Dataset size recommendations
            n_rows = self.df.shape[0]
            n_cols = self.df.shape[1]
            
            if n_rows < 1000:
                recommendations.append({
                    'category': 'small_dataset',
                    'priority': 'high',
                    'action': 'Optimize for small dataset',
                    'details': f"Dataset has only {n_rows} rows",
                    'specific_steps': [
                        "Use simple models to avoid overfitting",
                        "Apply strong regularization techniques",
                        "Use leave-one-out or k-fold cross-validation",
                        "Consider data augmentation if applicable"
                    ]
                })
            elif n_rows > 100000:
                recommendations.append({
                    'category': 'large_dataset',
                    'priority': 'medium',
                    'action': 'Optimize for large dataset',
                    'details': f"Dataset has {n_rows} rows",
                    'specific_steps': [
                        "Use efficient algorithms (SGD, online learning)",
                        "Consider sampling for initial model development",
                        "Use parallel processing and GPU acceleration",
                        "Apply incremental learning techniques"
                    ]
                })
            
            # High dimensionality recommendations
            if n_cols > n_rows:
                recommendations.append({
                    'category': 'high_dimensionality',
                    'priority': 'high',
                    'action': 'Address curse of dimensionality',
                    'details': f"More features ({n_cols}) than samples ({n_rows})",
                    'specific_steps': [
                        "Apply aggressive feature selection",
                        "Use regularization (L1/L2) heavily",
                        "Consider dimensionality reduction (PCA)",
                        "Use models that handle high dimensions well (SVM, Neural Networks)"
                    ]
                })
            
            # Multicollinearity recommendations
            multicollinearity_warnings = self.insights.get('relationships', {}).get('multicollinearity_warnings', [])
            
            if multicollinearity_warnings:
                recommendations.append({
                    'category': 'multicollinearity',
                    'priority': 'medium',
                    'action': 'Handle multicollinearity',
                    'details': f"Found {len(multicollinearity_warnings)} multicollinearity groups",
                    'specific_steps': [
                        "Remove highly correlated features",
                        "Use Ridge regression instead of OLS",
                        "Apply PCA to correlated feature groups",
                        "Use VIF to quantify multicollinearity"
                    ]
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'error',
                'priority': 'low',
                'action': 'Modeling analysis incomplete',
                'details': f"Error in analysis: {str(e)}"
            })
        
        return recommendations
    
    def _generate_data_collection_recommendations(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        try:
            missing_info = self.insights.get('data_composition', {}).get('missing_data', {})
            missing_percentage = missing_info.get('missing_percentage', 0)
            
            if missing_percentage > 20:
                recommendations.append({
                    'category': 'data_completeness',
                    'priority': 'high',
                    'action': 'Improve data collection completeness',
                    'details': f"Dataset is {missing_percentage:.1f}% incomplete",
                    'specific_steps': [
                        "Review data collection processes for missing value causes",
                        "Implement data validation at collection points",
                        "Consider mandatory fields for critical information",
                        "Set up monitoring for data quality metrics"
                    ]
                })
            
            data_quality_issues = self.insights.get('data_quality', [])
            
            if len(data_quality_issues) > 0:
                recommendations.append({
                    'category': 'data_quality',
                    'priority': 'medium',
                    'action': 'Address data quality issues',
                    'details': f"Found {len(data_quality_issues)} data quality issues",
                    'specific_steps': [
                        "Standardize data entry formats",
                        "Implement data validation rules",
                        "Train data collectors on quality standards",
                        "Set up automated quality checks"
                    ]
                })
            
            n_rows = self.df.shape[0]
            
            if n_rows < 100:
                recommendations.append({
                    'category': 'sample_size',
                    'priority': 'high',
                    'action': 'Increase sample size significantly',
                    'details': f"Only {n_rows} samples available",
                    'specific_steps': [
                        "Collect more data before modeling",
                        "Consider extending data collection period",
                        "Explore additional data sources",
                        "Use bootstrap sampling to understand variability"
                    ]
                })
            elif n_rows < 1000:
                recommendations.append({
                    'category': 'sample_size',
                    'priority': 'medium',
                    'action': 'Consider increasing sample size',
                    'details': f"Dataset has {n_rows} samples",
                    'specific_steps': [
                        "Assess if more data would improve model performance",
                        "Consider power analysis for statistical tests",
                        "Monitor model stability with current sample size"
                    ]
                })
            
            # Feature diversity recommendations
            if len(self.categorical_cols) == 0 and len(self.numeric_cols) > 0:
                recommendations.append({
                    'category': 'feature_diversity',
                    'priority': 'low',
                    'action': 'Consider collecting categorical features',
                    'details': "Dataset contains only numerical features",
                    'specific_steps': [
                        "Collect categorical variables for segmentation",
                        "Add metadata about data collection context",
                        "Consider binning numerical features for interpretability"
                    ]
                })
            elif len(self.numeric_cols) == 0 and len(self.categorical_cols) > 0:
                recommendations.append({
                    'category': 'feature_diversity',
                    'priority': 'low',
                    'action': 'Consider collecting numerical features',
                    'details': "Dataset contains only categorical features",
                    'specific_steps': [
                        "Collect quantitative measurements",
                        "Add continuous metrics for better model performance",
                        "Consider creating derived numerical features"
                    ]
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'error',
                'priority': 'low',
                'action': 'Data collection analysis incomplete',
                'details': f"Error in analysis: {str(e)}"
            })
        
        return recommendations
    
    def _generate_general_insights(self) -> List[Dict[str, Any]]:
        insights = []
        
        try:
            # Dataset overview insight
            n_rows, n_cols = self.df.shape
            insights.append({
                'category': 'dataset_overview',
                'insight': f"Dataset contains {n_rows:,} rows and {n_cols} columns",
                'interpretation': self._interpret_dataset_size(n_rows, n_cols),
                'implications': self._get_size_implications(n_rows, n_cols)
            })
            
            # Feature type distribution insight
            feature_dist = {
                'numerical': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols),
                'boolean': len(self.boolean_cols)
            }
            
            dominant_type = max(feature_dist, key=feature_dist.get)
            insights.append({
                'category': 'feature_distribution',
                'insight': f"Dataset is primarily {dominant_type} ({feature_dist[dominant_type]}/{n_cols} features)",
                'interpretation': f"This suggests the dataset is suitable for {self._get_suitable_analysis_type(dominant_type)}",
                'implications': self._get_feature_type_implications(dominant_type)
            })
            
            # Data completeness insight
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            insights.append({
                'category': 'data_completeness',
                'insight': f"Dataset is {data_density}% complete",
                'interpretation': self._interpret_data_completeness(data_density),
                'implications': self._get_completeness_implications(data_density)
            })
            
            # Target variable insight (if available)
            if self.target_column:
                target_insights = self.insights.get('target_insights', {})
                task_type = target_insights.get('task_type', 'unknown')
                
                insights.append({
                    'category': 'target_analysis',
                    'insight': f"Target variable suggests a {task_type} problem",
                    'interpretation': self._interpret_task_type(task_type, target_insights),
                    'implications': self._get_task_type_implications(task_type)
                })
            
            # Relationship complexity insight
            strong_correlations = self.insights.get('relationships', {}).get('strong_correlations', [])
            insights.append({
                'category': 'relationship_complexity',
                'insight': f"Found {len(strong_correlations)} strong feature relationships",
                'interpretation': self._interpret_relationship_complexity(len(strong_correlations)),
                'implications': self._get_relationship_implications(len(strong_correlations))
            })
            
            # Data quality insight
            warnings = self.insights.get('statistical_warnings', [])
            high_priority_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'high']
            
            insights.append({
                'category': 'data_quality',
                'insight': f"Identified {len(high_priority_warnings)} high-priority data issues",
                'interpretation': self._interpret_data_quality(len(high_priority_warnings)),
                'implications': self._get_quality_implications(len(high_priority_warnings))
            })
            
        except Exception as e:
            insights.append({
                'category': 'error',
                'insight': 'General insights analysis incomplete',
                'interpretation': f"Error in analysis: {str(e)}",
                'implications': ["Review data format and try again"]
            })
        
        return insights
    
    def _interpret_dataset_size(self, n_rows: int, n_cols: int) -> str:
        if n_rows < 100:
            return "Very small dataset - limited statistical power"
        elif n_rows < 1000:
            return "Small dataset - suitable for simple models"
        elif n_rows < 10000:
            return "Medium dataset - good for most ML algorithms"
        elif n_rows < 100000:
            return "Large dataset - suitable for complex models"
        else:
            return "Very large dataset - consider big data techniques"
    
    def _get_size_implications(self, n_rows: int, n_cols: int) -> List[str]:
        implications = []
        
        if n_rows < 1000:
            implications.extend([
                "Use simple models to avoid overfitting",
                "Apply strong cross-validation",
                "Consider collecting more data"
            ])
        elif n_rows > 100000:
            implications.extend([
                "Can use complex models safely",
                "Consider computational efficiency",
                "May benefit from sampling for development"
            ])
        
        if n_cols > n_rows:
            implications.append("High dimensional - use regularization or feature selection")
        
        return implications
    
    def _get_suitable_analysis_type(self, dominant_type: str) -> str:
        type_mapping = {
            'numerical': 'regression analysis and correlation studies',
            'categorical': 'classification and association analysis',
            'datetime': 'time series analysis and trend detection',
            'boolean': 'binary classification and flag analysis'
        }
        return type_mapping.get(dominant_type, 'general machine learning')
    
    def _get_feature_type_implications(self, dominant_type: str) -> List[str]:
        implications_mapping = {
            'numerical': [
                "Consider scaling and normalization",
                "Check for outliers and skewness",
                "Correlation analysis will be meaningful"
            ],
            'categorical': [
                "Encoding will be necessary for ML",
                "Check for high cardinality issues",
                "Association analysis recommended"
            ],
            'datetime': [
                "Extract temporal features",
                "Consider seasonality patterns",
                "Time-based validation splits needed"
            ],
            'boolean': [
                "Simple binary analysis applicable",
                "Check for class imbalance",
                "Consider as feature interactions"
            ]
        }
        return implications_mapping.get(dominant_type, ["General preprocessing needed"])
    
    def _interpret_data_completeness(self, data_density: float) -> str:
        if data_density >= 95:
            return "Excellent data quality with minimal missing values"
        elif data_density >= 85:
            return "Good data quality with some missing values"
        elif data_density >= 70:
            return "Moderate data quality - missing values need attention"
        else:
            return "Poor data quality - significant missing data issues"
    
    def _get_completeness_implications(self, data_density: float) -> List[str]:
        if data_density < 70:
            return [
                "Significant imputation will be needed",
                "Consider removing incomplete features",
                "Data collection process needs review"
            ]
        elif data_density < 90:
            return [
                "Moderate imputation required",
                "Check patterns in missing data",
                "Consider missingness as a feature"
            ]
        else:
            return [
                "Minimal preprocessing for missing values",
                "Simple imputation strategies sufficient"
            ]
    
    def _interpret_task_type(self, task_type: str, target_insights: Dict) -> str:
        if task_type == 'classification':
            n_classes = target_insights.get('class_balance', {}).get('n_classes', 0)
            if n_classes == 2:
                return "Binary classification problem - predict between two classes"
            else:
                return f"Multi-class classification problem - predict among {n_classes} classes"
        elif task_type == 'regression':
            return "Regression problem - predict continuous numerical values"
        else:
            return "Task type unclear - may need manual specification"
    
    def _get_task_type_implications(self, task_type: str) -> List[str]:
        if task_type == 'classification':
            return [
                "Use classification algorithms and metrics",
                "Check for class imbalance",
                "Consider stratified sampling"
            ]
        elif task_type == 'regression':
            return [
                "Use regression algorithms and metrics",
                "Check target distribution normality",
                "Consider transformation if needed"
            ]
        else:
            return ["Define problem type before modeling"]
    
    def _interpret_relationship_complexity(self, n_relationships: int) -> str:
        """Interpret relationship complexity"""
        if n_relationships == 0:
            return "Features appear independent - limited multicollinearity"
        elif n_relationships < 5:
            return "Few strong relationships - moderate feature interdependence"
        elif n_relationships < 15:
            return "Many relationships - high feature interdependence"
        else:
            return "Very complex relationships - potential multicollinearity issues"
    
    def _get_relationship_implications(self, n_relationships: int) -> List[str]:
        """Get implications of relationship complexity"""
        if n_relationships == 0:
            return [
                "Feature selection may be less critical",
                "Linear models should work well"
            ]
        elif n_relationships > 10:
            return [
                "Consider dimensionality reduction",
                "Check for multicollinearity",
                "Feature selection recommended"
            ]
        else:
            return [
                "Moderate feature engineering needed",
                "Monitor for overfitting"
            ]
    
    def _interpret_data_quality(self, n_issues: int) -> str:
        """Interpret data quality based on issues found"""
        if n_issues == 0:
            return "Excellent data quality - no critical issues found"
        elif n_issues < 3:
            return "Good data quality with minor issues"
        elif n_issues < 6:
            return "Moderate data quality - several issues need attention"
        else:
            return "Poor data quality - many critical issues identified"
    
    def _get_quality_implications(self, n_issues: int) -> List[str]:
        """Get implications of data quality"""
        if n_issues == 0:
            return ["Proceed with standard modeling approaches"]
        elif n_issues < 3:
            return ["Address minor issues before modeling"]
        else:
            return [
                "Significant data cleaning required",
                "Review data collection processes",
                "Consider data quality impact on model performance"
            ]
    
    def _safe_generate_business_insights(self):
        """Generate business-oriented insights safely"""
        try:
            business_insights = []
            
            # Data value insights
            business_insights.extend(self._generate_data_value_insights())
            
            # Operational insights
            business_insights.extend(self._generate_operational_insights())
            
            # Risk and compliance insights
            business_insights.extend(self._generate_risk_insights())
            
            # ROI and priority insights
            business_insights.extend(self._generate_roi_insights())
            
            self.insights['business_insights'] = business_insights
            
        except Exception as e:
            self.insights['business_insights'] = [f"Business insights generation failed: {str(e)}"]
    
    def _generate_data_value_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about data value and potential"""
        insights = []
        
        try:
            n_rows, n_cols = self.df.shape
            
            # Dataset completeness value
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            
            if data_density > 90:
                insights.append({
                    'category': 'data_asset_value',
                    'insight': 'High-quality data asset',
                    'business_impact': 'This dataset represents a valuable, clean data asset suitable for immediate analysis and modeling',
                    'action_items': [
                        'Prioritize this dataset for ML initiatives',
                        'Use as benchmark for data quality standards',
                        'Consider expanding similar data collection'
                    ],
                    'value_score': 'high'
                })
            elif data_density > 70:
                insights.append({
                    'category': 'data_asset_value',
                    'insight': 'Moderate-quality data asset',
                    'business_impact': 'Dataset has potential but requires investment in data quality improvement',
                    'action_items': [
                        'Budget for data cleaning initiatives',
                        'Implement data quality monitoring',
                        'Train staff on data collection best practices'
                    ],
                    'value_score': 'medium'
                })
            else:
                insights.append({
                    'category': 'data_asset_value',
                    'insight': 'Low-quality data asset',
                    'business_impact': 'Significant investment needed before this data can drive business value',
                    'action_items': [
                        'Major data quality initiative required',
                        'Review and redesign data collection processes',
                        'Consider cost-benefit of data remediation vs. new collection'
                    ],
                    'value_score': 'low'
                })
            
            # Sample size business impact
            if n_rows > 10000:
                insights.append({
                    'category': 'analytical_power',
                    'insight': 'Statistically powerful dataset',
                    'business_impact': 'Large sample size enables confident decision-making and reliable insights',
                    'action_items': [
                        'Leverage for high-stakes business decisions',
                        'Use for A/B testing and experimentation',
                        'Develop predictive models for operational use'
                    ],
                    'value_score': 'high'
                })
            elif n_rows < 100:
                insights.append({
                    'category': 'analytical_power',
                    'insight': 'Limited analytical power',
                    'business_impact': 'Small sample size limits confidence in insights and model reliability',
                    'action_items': [
                        'Use insights for exploration only',
                        'Invest in expanding data collection',
                        'Combine with external data sources if possible'
                    ],
                    'value_score': 'low'
                })
            
        except Exception as e:
            insights.append({
                'category': 'error',
                'insight': 'Data value analysis incomplete',
                'business_impact': f"Analysis error: {str(e)}"
            })
        
        return insights
    
    def _generate_operational_insights(self) -> List[Dict[str, Any]]:
        """Generate operational insights"""
        insights = []
        
        try:
            # Data processing complexity
            n_cols = self.df.shape[1]
            complexity_score = self._calculate_processing_complexity()
            
            if complexity_score > 0.7:
                insights.append({
                    'category': 'operational_complexity',
                    'insight': 'High data processing complexity',
                    'business_impact': 'Significant technical resources needed for data preparation and modeling',
                    'action_items': [
                        'Allocate experienced data science resources',
                        'Plan for extended project timeline',
                        'Consider MLOps infrastructure investment'
                    ],
                    'complexity_score': complexity_score
                })
            elif complexity_score < 0.3:
                insights.append({
                    'category': 'operational_complexity',
                    'insight': 'Low data processing complexity',
                    'business_impact': 'Dataset is ready for quick analysis and modeling with minimal preparation',
                    'action_items': [
                        'Fast-track for immediate business insights',
                        'Use as training ground for junior analysts',
                        'Implement automated processing pipelines'
                    ],
                    'complexity_score': complexity_score
                })
            
            # Feature engineering potential
            if len(self.datetime_cols) > 0:
                insights.append({
                    'category': 'feature_engineering_potential',
                    'insight': 'High temporal feature potential',
                    'business_impact': 'Datetime features enable time-based insights and forecasting capabilities',
                    'action_items': [
                        'Develop time series forecasting models',
                        'Create seasonal business insights',
                        'Implement trend monitoring systems'
                    ]
                })
            
            if len(self.categorical_cols) > 5:
                insights.append({
                    'category': 'segmentation_potential',
                    'insight': 'High customer/entity segmentation potential',
                    'business_impact': 'Rich categorical data enables detailed segmentation and personalization',
                    'action_items': [
                        'Develop customer segmentation strategies',
                        'Create personalized recommendation systems',
                        'Implement targeted marketing campaigns'
                    ]
                })
            
        except Exception as e:
            insights.append({
                'category': 'error',
                'insight': 'Operational analysis incomplete',
                'business_impact': f"Analysis error: {str(e)}"
            })
        
        return insights
    
    def _generate_risk_insights(self) -> List[Dict[str, Any]]:
        """Generate risk and compliance insights"""
        insights = []
        
        try:
            # Data quality risks
            high_priority_warnings = [w for w in self.insights.get('statistical_warnings', []) 
                                    if isinstance(w, dict) and w.get('severity') == 'high']
            
            if len(high_priority_warnings) > 3:
                insights.append({
                    'category': 'data_quality_risk',
                    'insight': 'High data quality risk identified',
                    'business_impact': 'Poor data quality could lead to incorrect business decisions and model failures',
                    'action_items': [
                        'Implement immediate data quality remediation',
                        'Establish data governance framework',
                        'Create data quality monitoring dashboard'
                    ],
                    'risk_level': 'high'
                })
            
            # Model reliability risks
            n_rows = self.df.shape[0]
            if n_rows < 1000:
                insights.append({
                    'category': 'model_reliability_risk',
                    'insight': 'Model reliability risk due to small sample size',
                    'business_impact': 'Models may not generalize well, leading to poor performance in production',
                    'action_items': [
                        'Use conservative model validation approaches',
                        'Implement robust monitoring in production',
                        'Plan for model retraining with more data'
                    ],
                    'risk_level': 'medium'
                })
            
            # Privacy and compliance considerations
            potential_pii_cols = self._identify_potential_pii_columns()
            if potential_pii_cols:
                insights.append({
                    'category': 'privacy_compliance_risk',
                    'insight': f'Potential PII detected in {len(potential_pii_cols)} columns',
                    'business_impact': 'May require special handling for privacy regulations (GDPR, CCPA)',
                    'action_items': [
                        'Review columns for actual PII content',
                        'Implement data anonymization if needed',
                        'Ensure compliance with privacy regulations',
                        'Document data lineage and usage'
                    ],
                    'risk_level': 'high',
                    'potential_pii_columns': potential_pii_cols
                })
            
        except Exception as e:
            insights.append({
                'category': 'error',
                'insight': 'Risk analysis incomplete',
                'business_impact': f"Analysis error: {str(e)}"
            })
        
        return insights
    
    def _generate_roi_insights(self) -> List[Dict[str, Any]]:
        """Generate ROI and priority insights"""
        insights = []
        
        try:
            # Quick wins identification
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            complexity_score = self._calculate_processing_complexity()
            
            if data_density > 85 and complexity_score < 0.4:
                insights.append({
                    'category': 'quick_wins',
                    'insight': 'High potential for quick wins',
                    'business_impact': 'Clean, simple dataset enables rapid time-to-value for analytics initiatives',
                    'action_items': [
                        'Prioritize for immediate analysis',
                        'Develop proof-of-concept models quickly',
                        'Use for demonstrating analytics ROI'
                    ],
                    'roi_potential': 'high',
                    'time_to_value': 'short'
                })
            
            # Long-term value assessment
            n_rows, n_cols = self.df.shape
            if n_rows > 10000 and n_cols > 10:
                insights.append({
                    'category': 'strategic_value',
                    'insight': 'High strategic data asset value',
                    'business_impact': 'Large, feature-rich dataset suitable for advanced analytics and AI initiatives',
                    'action_items': [
                        'Include in strategic data roadmap',
                        'Invest in advanced analytics capabilities',
                        'Consider as foundation for AI/ML center of excellence'
                    ],
                    'roi_potential': 'very_high',
                    'time_to_value': 'medium'
                })
            
            # Investment priority scoring
            priority_score = self._calculate_investment_priority()
            insights.append({
                'category': 'investment_priority',
                'insight': f'Investment priority score: {priority_score:.2f}/1.0',
                'business_impact': self._interpret_priority_score(priority_score),
                'action_items': self._get_priority_actions(priority_score),
                'priority_score': priority_score
            })
            
        except Exception as e:
            insights.append({
                'category': 'error',
                'insight': 'ROI analysis incomplete',
                'business_impact': f"Analysis error: {str(e)}"
            })
        
        return insights
    
    def _calculate_processing_complexity(self) -> float:
        """Calculate data processing complexity score (0-1)"""
        try:
            complexity_factors = []
            
            # Missing data complexity
            missing_pct = self.insights.get('data_composition', {}).get('missing_data', {}).get('missing_percentage', 0)
            complexity_factors.append(min(missing_pct / 50, 1.0))  # Normalize to 0-1
            
            # High cardinality complexity
            categorical_patterns = self.insights.get('feature_patterns', {}).get('categorical_patterns', {})
            high_cardinality_count = sum(1 for pattern in categorical_patterns.values() 
                                       if isinstance(pattern, dict) and pattern.get('high_cardinality', False))
            complexity_factors.append(min(high_cardinality_count / len(self.categorical_cols) if self.categorical_cols else 0, 1.0))
            
            # Outlier complexity
            warnings = self.insights.get('statistical_warnings', [])
            outlier_warnings = sum(1 for w in warnings if isinstance(w, dict) and w.get('type') == 'many_outliers')
            complexity_factors.append(min(outlier_warnings / len(self.numeric_cols) if self.numeric_cols else 0, 1.0))
            
            # Data quality complexity
            quality_issues = len([w for w in warnings if isinstance(w, dict) and w.get('severity') == 'high'])
            complexity_factors.append(min(quality_issues / 10, 1.0))
            
            return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.0
            
        except:
            return 0.5  # Default medium complexity
    
    def _identify_potential_pii_columns(self) -> List[str]:
        """Identify columns that might contain PII"""
        pii_indicators = [
            'name', 'email', 'phone', 'address', 'ssn', 'social', 'id', 'user',
            'customer', 'account', 'person', 'individual', 'contact', 'zip',
            'postal', 'credit', 'card', 'license', 'passport', 'birth', 'age'
        ]
        
        potential_pii = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in pii_indicators):
                potential_pii.append(col)
        
        return potential_pii
    
    def _calculate_investment_priority(self) -> float:
        """Calculate investment priority score (0-1)"""
        try:
            priority_factors = []
            
            # Data quality factor
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            priority_factors.append(data_density / 100)
            
            # Sample size factor
            n_rows = self.df.shape[0]
            if n_rows >= 10000:
                size_factor = 1.0
            elif n_rows >= 1000:
                size_factor = 0.8
            elif n_rows >= 100:
                size_factor = 0.5
            else:
                size_factor = 0.2
            priority_factors.append(size_factor)
            
            # Feature richness factor
            n_cols = self.df.shape[1]
            if n_cols >= 20:
                feature_factor = 1.0
            elif n_cols >= 10:
                feature_factor = 0.8
            elif n_cols >= 5:
                feature_factor = 0.6
            else:
                feature_factor = 0.4
            priority_factors.append(feature_factor)
            
            # Complexity factor (inverse - simpler is better for ROI)
            complexity = self._calculate_processing_complexity()
            priority_factors.append(1.0 - complexity)
            
            # Target availability factor
            target_factor = 1.0 if self.target_column else 0.7
            priority_factors.append(target_factor)
            
            return sum(priority_factors) / len(priority_factors)
            
        except:
            return 0.5  # Default medium priority
    
    def _interpret_priority_score(self, score: float) -> str:
        """Interpret priority score"""
        if score >= 0.8:
            return "Very high priority - Excellent ROI potential with minimal risk"
        elif score >= 0.6:
            return "High priority - Good ROI potential with manageable complexity"
        elif score >= 0.4:
            return "Medium priority - Moderate ROI potential, requires careful planning"
        else:
            return "Low priority - Limited ROI potential, high complexity or risk"
    
    def _get_priority_actions(self, score: float) -> List[str]:
        """Get actions based on priority score"""
        if score >= 0.8:
            return [
                "Fast-track for immediate implementation",
                "Allocate best resources to this project",
                "Use as showcase for analytics success"
            ]
        elif score >= 0.6:
            return [
                "Include in next quarter's roadmap",
                "Assign experienced team members",
                "Plan for production deployment"
            ]
        elif score >= 0.4:
            return [
                "Consider for future phases",
                "Address data quality issues first",
                "Evaluate cost-benefit carefully"
            ]
        else:
            return [
                "Defer until data quality improves",
                "Consider alternative data sources",
                "Focus on higher-priority datasets first"
            ]
    
    def _safe_generate_key_insights(self):
        """Generate key insights summary safely"""
        try:
            key_insights = []
            
            # Top 5 most important insights
            insights_candidates = []
            
            # Data quality insight
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            if data_density < 70:
                insights_candidates.append({
                    'insight': f"Critical: Dataset is only {data_density}% complete",
                    'priority': 10,
                    'category': 'data_quality',
                    'action_needed': True
                })
            elif data_density > 95:
                insights_candidates.append({
                    'insight': f"Excellent: Dataset is {data_density}% complete",
                    'priority': 7,
                    'category': 'data_quality',
                    'action_needed': False
                })
            
            # Target variable insight
            if self.target_column:
                target_insights = self.insights.get('target_insights', {})
                task_type = target_insights.get('task_type', 'unknown')
                
                if task_type == 'classification':
                    class_balance = target_insights.get('class_balance', {})
                    if class_balance.get('severity') == 'high':
                        insights_candidates.append({
                            'insight': f"Severe class imbalance detected (ratio: {class_balance.get('imbalance_ratio', 'N/A')})",
                            'priority': 9,
                            'category': 'target_analysis',
                            'action_needed': True
                        })
                    else:
                        insights_candidates.append({
                            'insight': f"Target variable suitable for {task_type} with {class_balance.get('n_classes', 'N/A')} classes",
                            'priority': 6,
                            'category': 'target_analysis',
                            'action_needed': False
                        })
                elif task_type == 'regression':
                    insights_candidates.append({
                        'insight': f"Target variable suitable for regression analysis",
                        'priority': 6,
                        'category': 'target_analysis',
                        'action_needed': False
                    })
            
            # Sample size insight
            n_rows = self.df.shape[0]
            if n_rows < 100:
                insights_candidates.append({
                    'insight': f"Very small dataset ({n_rows} rows) - Limited statistical power",
                    'priority': 8,
                    'category': 'sample_size',
                    'action_needed': True
                })
            elif n_rows > 10000:
                insights_candidates.append({
                    'insight': f"Large dataset ({n_rows:,} rows) - Strong statistical power",
                    'priority': 5,
                    'category': 'sample_size',
                    'action_needed': False
                })
            
            # Feature analysis insight
            total_features = len(self.numeric_cols) + len(self.categorical_cols) + len(self.datetime_cols) + len(self.boolean_cols)
            if total_features > n_rows:
                insights_candidates.append({
                    'insight': f"High dimensionality: {total_features} features vs {n_rows} samples",
                    'priority': 9,
                    'category': 'dimensionality',
                    'action_needed': True
                })
            elif total_features > 20:
                insights_candidates.append({
                    'insight': f"Feature-rich dataset: {total_features} features available",
                    'priority': 4,
                    'category': 'feature_richness',
                    'action_needed': False
                })
            
            # Missing data insight
            missing_info = self.insights.get('data_composition', {}).get('missing_data', {})
            high_missing_cols = missing_info.get('high_missing_columns', [])
            if len(high_missing_cols) > 0:
                insights_candidates.append({
                    'insight': f"{len(high_missing_cols)} columns have >50% missing values",
                    'priority': 8,
                    'category': 'missing_data',
                    'action_needed': True
                })
            
            # Correlation insight
            strong_correlations = self.insights.get('relationships', {}).get('strong_correlations', [])
            if len(strong_correlations) > 10:
                insights_candidates.append({
                    'insight': f"High feature interdependence: {len(strong_correlations)} strong correlations",
                    'priority': 7,
                    'category': 'relationships',
                    'action_needed': True
                })
            elif len(strong_correlations) > 0:
                insights_candidates.append({
                    'insight': f"{len(strong_correlations)} meaningful feature relationships identified",
                    'priority': 5,
                    'category': 'relationships',
                    'action_needed': False
                })
            
            # Data quality warnings insight
            warnings = self.insights.get('statistical_warnings', [])
            high_priority_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'high']
            if len(high_priority_warnings) > 5:
                insights_candidates.append({
                    'insight': f"{len(high_priority_warnings)} critical data quality issues identified",
                    'priority': 10,
                    'category': 'data_quality',
                    'action_needed': True
                })
            elif len(high_priority_warnings) > 0:
                insights_candidates.append({
                    'insight': f"{len(high_priority_warnings)} data quality issues need attention",
                    'priority': 7,
                    'category': 'data_quality',
                    'action_needed': True
                })
            
            # Business value insight
            priority_score = self._calculate_investment_priority()
            if priority_score > 0.8:
                insights_candidates.append({
                    'insight': f"High-value data asset (Priority Score: {priority_score:.2f})",
                    'priority': 6,
                    'category': 'business_value',
                    'action_needed': False
                })
            elif priority_score < 0.4:
                insights_candidates.append({
                    'insight': f"Low-value data asset (Priority Score: {priority_score:.2f})",
                    'priority': 8,
                    'category': 'business_value',
                    'action_needed': True
                })
            
            # Sort by priority and take top insights
            insights_candidates.sort(key=lambda x: x['priority'], reverse=True)
            key_insights = insights_candidates[:8]  # Top 8 insights
            
            # Add summary insight
            action_needed_count = sum(1 for insight in key_insights if insight['action_needed'])
            if action_needed_count > 0:
                summary_insight = {
                    'insight': f"Summary: {action_needed_count}/{len(key_insights)} insights require immediate action",
                    'priority': 5,
                    'category': 'summary',
                    'action_needed': action_needed_count > len(key_insights) // 2
                }
                key_insights.append(summary_insight)
            
            self.insights['key_insights'] = key_insights
            
        except Exception as e:
            self.insights['key_insights'] = [
                {
                    'insight': f"Key insights generation failed: {str(e)}",
                    'priority': 1,
                    'category': 'error',
                    'action_needed': True
                }
            ]
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("=" * 80)
            report_lines.append("AI-POWERED DATA ANALYSIS REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Dataset Overview
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 40)
            n_rows, n_cols = self.df.shape
            report_lines.append(f"Shape: {n_rows:,} rows × {n_cols} columns")
            
            feature_summary = []
            if self.numeric_cols:
                feature_summary.append(f"{len(self.numeric_cols)} numerical")
            if self.categorical_cols:
                feature_summary.append(f"{len(self.categorical_cols)} categorical")
            if self.datetime_cols:
                feature_summary.append(f"{len(self.datetime_cols)} datetime")
            if self.boolean_cols:
                feature_summary.append(f"{len(self.boolean_cols)} boolean")
            
            report_lines.append(f"Features: {', '.join(feature_summary)}")
            
            data_density = self.insights.get('data_composition', {}).get('data_density', 0)
            report_lines.append(f"Data Completeness: {data_density}%")
            report_lines.append("")
            
            # Key Insights
            report_lines.append("KEY INSIGHTS")
            report_lines.append("-" * 40)
            key_insights = self.insights.get('key_insights', [])
            for i, insight in enumerate(key_insights[:5], 1):
                if isinstance(insight, dict):
                    report_lines.append(f"{i}. {insight.get('insight', 'N/A')}")
            report_lines.append("")
            
            # Target Analysis (if available)
            if self.target_column:
                report_lines.append("TARGET VARIABLE ANALYSIS")
                report_lines.append("-" * 40)
                target_insights = self.insights.get('target_insights', {})
                task_type = target_insights.get('task_type', 'unknown')
                report_lines.append(f"Problem Type: {task_type.title()}")
                
                if task_type == 'classification':
                    class_balance = target_insights.get('class_balance', {})
                    report_lines.append(f"Classes: {class_balance.get('n_classes', 'N/A')}")
                    report_lines.append(f"Balance: {class_balance.get('status', 'N/A')}")
                elif task_type == 'regression':
                    stats = target_insights.get('statistics', {})
                    report_lines.append(f"Mean: {stats.get('mean', 'N/A')}")
                    report_lines.append(f"Std Dev: {stats.get('std', 'N/A')}")
                
                report_lines.append("")
            
            # Top Recommendations
            report_lines.append("TOP RECOMMENDATIONS")
            report_lines.append("-" * 40)
            ai_recommendations = self.insights.get('ai_recommendations', {})
            
            # Get top recommendations from each category
            top_recommendations = []
            for category, recs in ai_recommendations.items():
                if isinstance(recs, list) and len(recs) > 0:
                    high_priority_recs = [r for r in recs if isinstance(r, dict) and r.get('priority') == 'high']
                    if high_priority_recs:
                        top_recommendations.extend(high_priority_recs[:1])  # Top 1 from each category
            
            for i, rec in enumerate(top_recommendations[:5], 1):
                if isinstance(rec, dict):
                    report_lines.append(f"{i}. {rec.get('action', 'N/A')} ({rec.get('category', 'N/A')})")
            
            if not top_recommendations:
                report_lines.append("No high-priority recommendations identified.")
            
            report_lines.append("")
            
            # Data Quality Summary
            report_lines.append("DATA QUALITY SUMMARY")
            report_lines.append("-" * 40)
            warnings = self.insights.get('statistical_warnings', [])
            high_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'high']
            medium_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'medium']
            
            report_lines.append(f"Critical Issues: {len(high_warnings)}")
            report_lines.append(f"Medium Issues: {len(medium_warnings)}")
            report_lines.append(f"Overall Quality: {'Excellent' if len(high_warnings) == 0 else 'Needs Attention'}")
            report_lines.append("")
            
            # Business Impact
            report_lines.append("💼 BUSINESS IMPACT ASSESSMENT")
            report_lines.append("-" * 40)
            priority_score = self._calculate_investment_priority()
            report_lines.append(f"Investment Priority: {priority_score:.2f}/1.0")
            report_lines.append(f"Value Assessment: {self._interpret_priority_score(priority_score)}")
            report_lines.append("")
            
            # Footer
            report_lines.append("=" * 80)
            report_lines.append("Report generated by RobustAutomatedInsightGenerator")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Summary report generation failed: {str(e)}"
    
    def export_insights_to_dict(self) -> Dict[str, Any]:
        """Export all insights to a clean dictionary format"""
        try:
            export_dict = {
                'metadata': {
                    'analysis_timestamp': pd.Timestamp.now().isoformat(),
                    'dataset_shape': self.df.shape,
                    'target_column': self.target_column,
                    'feature_types': {
                        'numerical': self.numeric_cols,
                        'categorical': self.categorical_cols,
                        'datetime': self.datetime_cols,
                        'boolean': self.boolean_cols
                    }
                },
                'insights': self.insights,
                'summary_report': self.generate_summary_report()
            }
            
            return export_dict
            
        except Exception as e:
            return {
                'error': f"Export failed: {str(e)}",
                'partial_insights': self.insights
            }
        

class DataQualityReportGenerator:
    """
    Professional Data Quality Report Generator
    
    Generates comprehensive dataset health and reliability reports
    suitable for presentation to stakeholders and management.
    """
    
    def __init__(self, dataframe, dataset_name="Dataset"):
        """
        Initialize the Data Quality Report Generator
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The dataset to analyze
        dataset_name : str
            Name of the dataset for report identification
        """
        self.df = dataframe.copy()
        self.dataset_name = dataset_name
        self.report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def generate_comprehensive_report(self):
        """
        Generate comprehensive data quality report
        
        Returns:
        --------
        dict: Complete data quality assessment
        """
        print("Generating Professional Data Quality Report...")
        print("=" * 60)
        
        # Core dataset information
        basic_info = self._get_basic_dataset_info()
        
        # Missing values analysis
        missing_analysis = self._analyze_missing_values()
        
        # Zero values analysis
        zero_analysis = self._analyze_zero_values()
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates()
        
        # Data type consistency
        type_consistency = self._analyze_data_type_consistency()
        
        # Cardinality analysis
        cardinality_analysis = self._analyze_cardinality()
        
        # Constant features detection
        constant_features = self._detect_constant_features()
        
        # Outlier detection
        outlier_analysis = self._detect_outliers()
        
        # Overall health score
        health_score = self._calculate_health_score(
            missing_analysis, duplicate_analysis, 
            constant_features, type_consistency
        )
        
        # Compile comprehensive report
        comprehensive_report = {
            'report_metadata': {
                'dataset_name': self.dataset_name,
                'generated_on': self.report_timestamp,
                'total_rows': basic_info['total_rows'],
                'total_columns': basic_info['total_columns'],
                'overall_health_score': health_score['score'],
                'health_grade': health_score['grade']
            },
            'basic_information': basic_info,
            'missing_values_analysis': missing_analysis,
            'zero_values_analysis': zero_analysis,
            'duplicate_analysis': duplicate_analysis,
            'data_type_consistency': type_consistency,
            'cardinality_analysis': cardinality_analysis,
            'constant_features': constant_features,
            'outlier_analysis': outlier_analysis,
            'health_assessment': health_score
        }
        
        # Display professional report
        self._display_professional_report(comprehensive_report)
        
        return comprehensive_report
    
    def _get_basic_dataset_info(self):
        """Get basic dataset information"""
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': round(memory_usage, 2),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns),
            'column_names': list(self.df.columns)
        }
    
    def _analyze_missing_values(self):
        """Comprehensive missing values analysis"""
        missing_stats = []
        total_cells = len(self.df) * len(self.df.columns)
        
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            missing_stats.append({
                'column': column,
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_percentage, 2),
                'data_type': str(self.df[column].dtype),
                'severity': 'High' if missing_percentage > 50 else 'Medium' if missing_percentage > 20 else 'Low'
            })
        
        # Sort by missing percentage (descending)
        missing_stats.sort(key=lambda x: x['missing_percentage'], reverse=True)
        
        total_missing = sum([stat['missing_count'] for stat in missing_stats])
        overall_missing_rate = (total_missing / total_cells) * 100
        
        return {
            'column_wise_missing': missing_stats,
            'total_missing_values': int(total_missing),
            'overall_missing_rate': round(overall_missing_rate, 2),
            'columns_with_missing': len([s for s in missing_stats if s['missing_count'] > 0]),
            'completely_empty_columns': [s['column'] for s in missing_stats if s['missing_percentage'] == 100]
        }
    
    def _analyze_zero_values(self):
        """Analyze zero values in numeric columns"""
        zero_stats = []
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            zero_count = (self.df[column] == 0).sum()
            zero_percentage = (zero_count / len(self.df)) * 100
            
            zero_stats.append({
                'column': column,
                'zero_count': int(zero_count),
                'zero_percentage': round(zero_percentage, 2),
                'severity': 'High' if zero_percentage > 70 else 'Medium' if zero_percentage > 30 else 'Low'
            })
        
        # Sort by zero percentage (descending)
        zero_stats.sort(key=lambda x: x['zero_percentage'], reverse=True)
        
        return {
            'column_wise_zeros': zero_stats,
            'zero_heavy_columns': [s['column'] for s in zero_stats if s['zero_percentage'] > 50],
            'analyzed_numeric_columns': len(numeric_columns)
        }
    
    def _analyze_duplicates(self):
        """Comprehensive duplicate analysis"""
        # Duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        duplicate_row_percentage = (duplicate_rows / len(self.df)) * 100
        
        # Duplicate columns (by content)
        duplicate_columns = []
        columns = list(self.df.columns)
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if self.df[col1].equals(self.df[col2]):
                    duplicate_columns.append((col1, col2))
        
        # Identify potential duplicate records
        duplicate_subsets = []
        if len(self.df.columns) > 1:
            # Check for duplicates excluding the last column (common ID scenario)
            subset_cols = list(self.df.columns[:-1])
            subset_duplicates = self.df.duplicated(subset=subset_cols).sum()
            if subset_duplicates > duplicate_rows:
                duplicate_subsets.append({
                    'subset': subset_cols,
                    'duplicate_count': int(subset_duplicates),
                    'description': 'Duplicates excluding last column'
                })
        
        return {
            'total_duplicate_rows': int(duplicate_rows),
            'duplicate_row_percentage': round(duplicate_row_percentage, 2),
            'duplicate_columns': duplicate_columns,
            'total_duplicate_column_pairs': len(duplicate_columns),
            'duplicate_subsets': duplicate_subsets,
            'unique_rows': len(self.df) - duplicate_rows
        }
    
    def _analyze_data_type_consistency(self):
        """Analyze data type consistency and detect mixed types"""
        type_issues = []
        
        for column in self.df.columns:
            col_data = self.df[column].dropna()
            
            if col_data.dtype == 'object':
                # Check for mixed numeric and string data
                numeric_count = 0
                string_count = 0
                
                for value in col_data.head(100):  # Sample first 100 non-null values
                    try:
                        float(value)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        string_count += 1
                
                if numeric_count > 0 and string_count > 0:
                    type_issues.append({
                        'column': column,
                        'issue': 'Mixed numeric and text data',
                        'numeric_count': numeric_count,
                        'string_count': string_count,
                        'severity': 'High'
                    })
        
        # Check for potential datetime columns stored as strings
        potential_datetime_cols = []
        for column in self.df.select_dtypes(include=['object']).columns:
            sample_values = self.df[column].dropna().head(10)
            datetime_like_count = 0
            
            for value in sample_values:
                if isinstance(value, str) and len(value) > 8:
                    # Simple heuristic for datetime-like strings
                    if any(char in str(value) for char in ['-', '/', ':']):
                        datetime_like_count += 1
            
            if datetime_like_count >= len(sample_values) * 0.7:
                potential_datetime_cols.append(column)
        
        return {
            'mixed_type_columns': type_issues,
            'potential_datetime_columns': potential_datetime_cols,
            'total_type_issues': len(type_issues),
            'data_type_summary': {str(dtype): len(self.df.select_dtypes(include=[dtype]).columns) 
                                for dtype in self.df.dtypes.unique()}
        }
    
    def _analyze_cardinality(self):
        """Analyze feature cardinality (uniqueness)"""
        cardinality_stats = []
        
        for column in self.df.columns:
            unique_count = self.df[column].nunique()
            unique_percentage = (unique_count / len(self.df)) * 100
            
            # Classify cardinality
            if unique_count == len(self.df):
                cardinality_type = "Unique (ID-like)"
            elif unique_percentage > 95:
                cardinality_type = "Very High"
            elif unique_percentage > 50:
                cardinality_type = "High"
            elif unique_percentage > 10:
                cardinality_type = "Medium"
            else:
                cardinality_type = "Low"
            
            cardinality_stats.append({
                'column': column,
                'unique_count': int(unique_count),
                'unique_percentage': round(unique_percentage, 2),
                'cardinality_type': cardinality_type,
                'data_type': str(self.df[column].dtype)
            })
        
        # Sort by unique percentage (descending)
        cardinality_stats.sort(key=lambda x: x['unique_percentage'], reverse=True)
        
        return {
            'column_wise_cardinality': cardinality_stats,
            'high_cardinality_columns': [s['column'] for s in cardinality_stats if s['cardinality_type'] in ['Very High', 'Unique (ID-like)']],
            'potential_id_columns': [s['column'] for s in cardinality_stats if s['cardinality_type'] == 'Unique (ID-like)']
        }
    
    def _detect_constant_features(self):
        """Detect constant features (single unique value)"""
        constant_features = []
        
        for column in self.df.columns:
            unique_count = self.df[column].nunique()
            if unique_count <= 1:
                constant_value = self.df[column].iloc[0] if len(self.df) > 0 else None
                constant_features.append({
                    'column': column,
                    'constant_value': constant_value,
                    'data_type': str(self.df[column].dtype)
                })
        
        return {
            'constant_columns': constant_features,
            'total_constant_features': len(constant_features),
            'impact_assessment': 'High - These columns provide no information for analysis or modeling'
        }
    
    def _detect_outliers(self):
        """Detect outliers using IQR method for numeric columns"""
        outlier_stats = []
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = self.df[column].dropna()
            
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(col_data)) * 100
                
                outlier_stats.append({
                    'column': column,
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': round(outlier_percentage, 2),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4),
                    'min_value': round(col_data.min(), 4),
                    'max_value': round(col_data.max(), 4),
                    'severity': 'High' if outlier_percentage > 10 else 'Medium' if outlier_percentage > 5 else 'Low'
                })
        
        # Sort by outlier percentage (descending)
        outlier_stats.sort(key=lambda x: x['outlier_percentage'], reverse=True)
        
        return {
            'column_wise_outliers': outlier_stats,
            'high_outlier_columns': [s['column'] for s in outlier_stats if s['severity'] == 'High'],
            'total_analyzed_columns': len(numeric_columns)
        }
    
    def _calculate_health_score(self, missing_analysis, duplicate_analysis, constant_features, type_consistency):
        """Calculate overall dataset health score"""
        score = 100
        issues = []
        
        # Deduct for missing values
        missing_penalty = min(missing_analysis['overall_missing_rate'] * 2, 30)
        score -= missing_penalty
        if missing_penalty > 0:
            issues.append(f"Missing values penalty: -{missing_penalty:.1f}")
        
        # Deduct for duplicates
        duplicate_penalty = min(duplicate_analysis['duplicate_row_percentage'] * 1.5, 20)
        score -= duplicate_penalty
        if duplicate_penalty > 0:
            issues.append(f"Duplicate rows penalty: -{duplicate_penalty:.1f}")
        
        # Deduct for constant features
        constant_penalty = len(constant_features['constant_columns']) * 5
        score -= constant_penalty
        if constant_penalty > 0:
            issues.append(f"Constant features penalty: -{constant_penalty}")
        
        # Deduct for type consistency issues
        type_penalty = len(type_consistency['mixed_type_columns']) * 10
        score -= type_penalty
        if type_penalty > 0:
            issues.append(f"Type consistency penalty: -{type_penalty}")
        
        score = max(0, score)
        
        # Assign grade
        if score >= 90:
            grade = "A - Excellent"
        elif score >= 80:
            grade = "B - Good"
        elif score >= 70:
            grade = "C - Fair"
        elif score >= 60:
            grade = "D - Poor"
        else:
            grade = "F - Critical Issues"
        
        return {
            'score': round(score, 1),
            'grade': grade,
            'penalty_breakdown': issues,
            'recommendation': self._get_health_recommendation(score)
        }
    
    def _get_health_recommendation(self, score):
        """Get recommendation based on health score"""
        if score >= 90:
            return "Dataset is in excellent condition. Ready for analysis and modeling."
        elif score >= 80:
            return "Dataset is in good condition. Minor cleaning may improve quality."
        elif score >= 70:
            return "Dataset has moderate issues. Data cleaning recommended before analysis."
        elif score >= 60:
            return "Dataset has significant quality issues. Extensive cleaning required."
        else:
            return "Dataset has critical quality issues. Major remediation needed before use."
    
    def _display_professional_report(self, report):
        """Display professional formatted report"""
        print("\n" + "="*80)
        print(f"PROFESSIONAL DATA QUALITY ASSESSMENT REPORT")
        print("="*80)
        
        # Header Information
        metadata = report['report_metadata']
        print(f"\nREPORT METADATA")
        print("-" * 40)
        print(f"Dataset Name: {metadata['dataset_name']}")
        print(f"Generated On: {metadata['generated_on']}")
        print(f"Dataset Size: {metadata['total_rows']:,} rows × {metadata['total_columns']} columns")
        print(f"Overall Health Score: {metadata['overall_health_score']}/100 ({metadata['health_grade']})")
        
        # Executive Summary
        print(f"\nEXECUTIVE SUMMARY")
        print("-" * 40)
        health = report['health_assessment']
        print(f"Health Score: {health['score']}/100")
        print(f"Grade: {health['grade']}")
        print(f"Recommendation: {health['recommendation']}")
        
        # Missing Values Summary
        missing = report['missing_values_analysis']
        print(f"\nMISSING VALUES ASSESSMENT")
        print("-" * 40)
        print(f"Overall Missing Rate: {missing['overall_missing_rate']}%")
        print(f"Columns with Missing Data: {missing['columns_with_missing']}/{metadata['total_columns']}")
        if missing['completely_empty_columns']:
            print(f"Completely Empty Columns: {len(missing['completely_empty_columns'])}")
        
        # Top 5 columns with highest missing rates
        if missing['column_wise_missing']:
            print("\nTop 5 Columns by Missing Rate:")
            for i, col_stat in enumerate(missing['column_wise_missing'][:5]):
                if col_stat['missing_percentage'] > 0:
                    print(f"  {i+1}. {col_stat['column']}: {col_stat['missing_percentage']}% ({col_stat['severity']} severity)")
        
        # Duplicate Analysis
        duplicates = report['duplicate_analysis']
        print(f"\nDUPLICATE ANALYSIS")
        print("-" * 40)
        print(f"Duplicate Rows: {duplicates['total_duplicate_rows']:,} ({duplicates['duplicate_row_percentage']}%)")
        print(f"Duplicate Column Pairs: {duplicates['total_duplicate_column_pairs']}")
        print(f"Unique Rows: {duplicates['unique_rows']:,}")
        
        # Data Type Issues
        types = report['data_type_consistency']
        print(f"\DATA TYPE CONSISTENCY")
        print("-" * 40)
        print(f"Mixed Type Issues: {types['total_type_issues']}")
        if types['potential_datetime_columns']:
            print(f"Potential DateTime Columns: {len(types['potential_datetime_columns'])}")
        
        # Cardinality Issues
        cardinality = report['cardinality_analysis']
        print(f"\nCARDINALITY ANALYSIS")
        print("-" * 40)
        print(f"High Cardinality Columns: {len(cardinality['high_cardinality_columns'])}")
        print(f"Potential ID Columns: {len(cardinality['potential_id_columns'])}")
        
        # Constant Features
        constants = report['constant_features']
        print(f"\nCONSTANT FEATURES")
        print("-" * 40)
        print(f"Constant Columns: {constants['total_constant_features']}")
        if constants['constant_columns']:
            print("Constant Columns Found:")
            for const in constants['constant_columns']:
                print(f"  • {const['column']} (value: {const['constant_value']})")
        
        # Outlier Summary
        outliers = report['outlier_analysis']
        print(f"\nOUTLIER DETECTION SUMMARY")
        print("-" * 40)
        high_outlier_cols = [col for col in outliers['column_wise_outliers'] if col['severity'] == 'High']
        print(f"Columns with High Outlier Rate: {len(high_outlier_cols)}")
        print(f"Numeric Columns Analyzed: {outliers['total_analyzed_columns']}")
        
        # Recommendations
        print(f"\nKEY RECOMMENDATIONS")
        print("-" * 40)
        recommendations = []
        
        if missing['overall_missing_rate'] > 20:
            recommendations.append("• Address high missing value rate through imputation or removal")
        
        if duplicates['duplicate_row_percentage'] > 5:
            recommendations.append("• Remove duplicate rows to improve data quality")
        
        if constants['total_constant_features'] > 0:
            recommendations.append("• Drop constant features as they provide no analytical value")
        
        if types['total_type_issues'] > 0:
            recommendations.append("• Fix data type inconsistencies for better analysis")
        
        if len(cardinality['potential_id_columns']) > 1:
            recommendations.append("• Review multiple ID-like columns for redundancy")
        
        if not recommendations:
            recommendations.append("• Dataset is in good condition, ready for analysis")
        
        for rec in recommendations:
            print(rec)
        
        print("\n" + "="*80)
        print("Report Generation Complete")
        print("="*80)
    
    def export_detailed_report_to_csv(self, filename=None):
        """Export detailed report to CSV format"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_quality_report_{timestamp}.csv"
        
        report = self.generate_comprehensive_report()
        
        # Create detailed CSV export
        export_data = []
        
        # Missing values details
        for col_stat in report['missing_values_analysis']['column_wise_missing']:
            export_data.append({
                'Assessment_Type': 'Missing_Values',
                'Column': col_stat['column'],
                'Issue_Count': col_stat['missing_count'],
                'Issue_Percentage': col_stat['missing_percentage'],
                'Severity': col_stat['severity'],
                'Data_Type': col_stat['data_type'],
                'Details': f"Missing: {col_stat['missing_count']} values"
            })
        
        # Zero values details
        for col_stat in report['zero_values_analysis']['column_wise_zeros']:
            export_data.append({
                'Assessment_Type': 'Zero_Values',
                'Column': col_stat['column'],
                'Issue_Count': col_stat['zero_count'],
                'Issue_Percentage': col_stat['zero_percentage'],
                'Severity': col_stat['severity'],
                'Data_Type': 'numeric',
                'Details': f"Zero values: {col_stat['zero_count']}"
            })
        
        # Outlier details
        for col_stat in report['outlier_analysis']['column_wise_outliers']:
            export_data.append({
                'Assessment_Type': 'Outliers',
                'Column': col_stat['column'],
                'Issue_Count': col_stat['outlier_count'],
                'Issue_Percentage': col_stat['outlier_percentage'],
                'Severity': col_stat['severity'],
                'Data_Type': 'numeric',
                'Details': f"Outliers: {col_stat['outlier_count']} (bounds: {col_stat['lower_bound']:.2f} to {col_stat['upper_bound']:.2f})"
            })
        
        # Cardinality details
        for col_stat in report['cardinality_analysis']['column_wise_cardinality']:
            export_data.append({
                'Assessment_Type': 'Cardinality',
                'Column': col_stat['column'],
                'Issue_Count': col_stat['unique_count'],
                'Issue_Percentage': col_stat['unique_percentage'],
                'Severity': 'High' if col_stat['cardinality_type'] in ['Very High', 'Unique (ID-like)'] else 'Low',
                'Data_Type': col_stat['data_type'],
                'Details': f"Unique values: {col_stat['unique_count']} ({col_stat['cardinality_type']})"
            })
        
        # Convert to DataFrame and save
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(filename, index=False)
        
        print(f"\nDetailed report exported to: {filename}")
        return filename

# Usage Example and Testing Function
@st.cache_data(show_spinner="Generating data quality report...")
def generate_data_quality_report(dataframe, dataset_name="Dataset", export_csv=True):
    """
    Main function to generate data quality report
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        The dataset to analyze
    dataset_name : str
        Name of the dataset
    export_csv : bool
        Whether to export detailed CSV report
    
    Returns:
    --------
    dict: Comprehensive data quality report
    """
    
    # Initialize the report generator
    report_generator = DataQualityReportGenerator(dataframe, dataset_name)
    
    # Generate comprehensive report
    quality_report = report_generator.generate_comprehensive_report()
    
    # Export CSV if requested
    if export_csv:
        csv_filename = report_generator.export_detailed_report_to_csv()
        quality_report['exported_csv'] = csv_filename
    
    return quality_report

