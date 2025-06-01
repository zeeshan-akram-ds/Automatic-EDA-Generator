import importlib
import core.eda_engine as eda_engine
importlib.reload(eda_engine)

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import time
import io

st.title("Exploratory Data Analysis (EDA)")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Sidebar for target column
st.sidebar.header("Target Column")
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Temporary load to populate sidebar
    target = st.sidebar.selectbox(
        "Select the target column (optional):",
        options=[None] + df.columns.tolist(),
        help="Choose a target for coloring plots (categorical or low-cardinality numeric recommended)."
    )
    if target and pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 15:
        st.sidebar.warning("High-cardinality numeric target detected. It will be binned in plots.")
else:
    target = None
# If file is uploaded
if uploaded_file:
    df, error = eda_engine.load_and_validate_csv(uploaded_file)

    # If loading error
    if error:
        st.error(error)

    # If loaded successfully
    elif df is not None:
        st.success("File loaded successfully!")
        
        # Preview
        st.write("### Preview of Data")
        st.dataframe(df.head())
        st.write("**Shape:**", df.shape)

        # Tabs for Summary and Visuals
        tab1, tab2 = st.tabs(["Dataset Summary", "Visuals"])

        with tab1:
            st.subheader("Dataset Overview")
            overview, overview_error = eda_engine.get_dataset_overview(df)

            if overview_error:
                st.error(overview_error)
            else:
                st.markdown("#### Basic Dataset Info")
                st.write("**Shape (rows, columns):**", overview["Shape (rows, columns)"])

                # Column summary table
                column_summary_df = pd.DataFrame(overview["Column Summary"])
                st.dataframe(column_summary_df, use_container_width=True)

                # Duplicate row info
                st.markdown(f"**Duplicate Rows:** `{overview['Duplicate Rows']}`")

        with tab2:
            st.subheader("Visualizations")

            # Inner tabs for organization
            uni_tab, bi_tab, multi_tab, adv_tab = st.tabs([
                "Univariate", "Bivariate", "Multivariate", "Advanced"
            ])
        
            with uni_tab:
                st.markdown("### Univariate Analysis")

                selected_col = st.selectbox("Select a column to visualize", df.columns)

                result = eda_engine.plot_univariate(df, selected_col)

                if "too_many_categories" in result:
                    st.warning(result["message"])
                elif "unsupported" in result:
                    st.error(result["message"])
                else:
                    st.markdown(f"#### Univariate Plot for `{selected_col}`")

                    if result["type"] == "numeric":
                        st.pyplot(result["fig"])
                    elif result["type"] == "categorical":
                        if isinstance(result["fig"], plt.Figure):
                            st.pyplot(result["fig"])
                        else:
                            st.plotly_chart(result["fig"], use_container_width=True)

            with bi_tab:
                st.subheader("Bivariate Analysis")

                analysis_tab = st.tabs(["Categorical vs Target", "Numeric vs Target", "Target Analysis"])

                # -----------------------------
                # Categorical vs Target
                # -----------------------------
                with analysis_tab[0]:
                    st.markdown("### Categorical vs Target")
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    cat_cols = [col for col in cat_cols if col != target]

                    if not cat_cols:
                        st.info("No categorical columns available other than the target.")
                    else:
                        selected_col = st.selectbox("Choose a categorical column to compare with target:", cat_cols)
                        st.info("Recommended: Works best when target is classification.")

                        if st.button("Plot Countplot"):
                            fig = eda_engine.plot_cat_vs_target(df, selected_col, target)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                # -----------------------------
                # Numeric vs Target
                # -----------------------------
                with analysis_tab[1]:
                    st.markdown("### Numeric vs Target")
                    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    num_cols = [col for col in num_cols if col != target]

                    if not num_cols:
                        st.info("No numeric columns available other than the target.")
                    else:
                        chart_type = st.selectbox("Select Chart Type:", ["Boxplot", "Violinplot", "Swarmplot"])
                        selected_col = st.selectbox("Choose a numeric column to compare with target:", num_cols)

                        if st.button("Plot Numeric Chart"):
                            fig = eda_engine.plot_num_vs_target(df, selected_col, target, chart_type)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                # -----------------------------
                # Target Analysis
                # -----------------------------
                with analysis_tab[2]:
                    st.markdown("### Target Distribution & Insights")

                    if st.button("Show Target Class Balance"):
                        pie, bar = eda_engine.plot_class_balance(df, target)
                        col1, col2 = st.columns(2)
                        if pie:
                            col1.plotly_chart(pie, use_container_width=True)
                        if bar:
                            col2.plotly_chart(bar, use_container_width=True)
                    st.markdown("---")

                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    if target in df.columns:
                        if df[target].dtype in ["int64", "float64"]:
                            cat_cols = [col for col in cat_cols if col != target]
                            st.markdown("#### Mean Target Value by Category")
                            selected_cat = st.selectbox("Choose categorical column:", cat_cols)

                            if st.button("Plot Mean Target per Category"):
                                fig = eda_engine.plot_mean_target_by_category(df, selected_cat, target)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

            with multi_tab:
                st.subheader("Multivariate Analysis")

                multi_inner_tabs = st.tabs(["Numeric vs Numeric", "Categorical vs Categorical", "Numeric vs Categorical", "Custom Relationships","AI Advanced Analysis"])

                # 1️NUMERIC vs NUMERIC
                with multi_inner_tabs[0]:
                    st.subheader("Pairwise Numeric Relationships")

                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

                    if len(numeric_cols) < 2:
                        st.warning("At least two numeric columns are required for pairwise analysis.")
                    else:
                        with st.form(key="pairwise_form"):
                            st.markdown("### Settings")
                            target_col = st.selectbox(
                                "Target Column (optional)",
                                [None] + df.columns.tolist(),
                                index=0,
                                key="pairwise_target"
                            )
                            selected_cols = st.multiselect(
                                "Select Numeric Columns",
                                options=numeric_cols,
                                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                            )
                            plot_type = st.selectbox(
                                "Plot Type",
                                ["all", "scatter", "correlation", "hexbin", "joint"]
                            )
                            sample_size = st.slider(
                                "Sample Size (for plotting)",
                                min_value=100,
                                max_value=min(len(df), 5000),
                                value=min(1000, len(df)),
                                step=100
                            )
                            show_trendline = st.checkbox("Show Trendline", value=False)
                            log_x = st.checkbox("Log Scale X-Axis", value=False)
                            log_y = st.checkbox("Log Scale Y-Axis", value=False)
                            show_outliers = st.checkbox("Highlight Outliers", value=False)
                            hexbin_bins = st.slider(
                                "Hexbin Bins",
                                min_value=10,
                                max_value=100,
                                value=30,
                                step=5
                            )
                            color_scale = st.selectbox(
                                "Color Scale",
                                ["Viridis", "Plasma", "Cividis", "Inferno", "Jet"]
                            )
                            submit_button = st.form_submit_button("Generate Visualizations")

                        if submit_button:
                            if len(selected_cols) < 2:
                                st.warning("Please select at least two numeric columns.")
                            else:
                                with st.spinner("Generating plots..."):
                                    progress_bar = st.progress(0)
                                    try:
                                        figs = eda_engine.explore_pairwise_relationships(
                                            df=df,
                                            target_col=target_col,
                                            cols=selected_cols,
                                            plot_type=plot_type,
                                            sample_size=sample_size,
                                            show_trendline=show_trendline,
                                            fig_size=(800, 600),
                                            correlation_method="pearson",
                                            color_scale=color_scale,
                                            log_x=log_x,
                                            log_y=log_y,
                                            hexbin_bins=hexbin_bins,
                                            show_outliers=show_outliers,
                                            return_figs=True
                                        )
                                        if figs:
                                            st.success(f"Generated {len(figs)} plots.")
                                            for i, fig in enumerate(figs):
                                                st.plotly_chart(fig, use_container_width=True)
                                                progress_bar.progress((i + 1) / len(figs))
                                        else:
                                            st.warning("No plots generated. Please check your selections.")
                                    except Exception as e:
                                        st.error(f"Error generating plots: {str(e)}")
                with multi_inner_tabs[1]:
                    st.subheader("Categorical Variable Relationships")

                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    if target:
                        cat_cols = [col for col in cat_cols if col != target]

                    if len(cat_cols) < 2:
                        st.warning("At least two categorical columns are required for analysis.")
                    else:
                        with st.form(key="cat_vs_cat_form"):
                            st.markdown("### Settings")
                            cat_col1 = st.selectbox(
                                "Select First Categorical Column",
                                options=cat_cols,
                                key="cat_col1"
                            )
                            cat_col2 = st.selectbox(
                                "Select Second Categorical Column",
                                options=[col for col in cat_cols if col != cat_col1],
                                key="cat_col2"
                            )
                            target_col = st.selectbox(
                                "Target Column (optional)",
                                [None] + df.columns.tolist(),
                                index=0,
                                key="cat_target"
                            )
                            plot_type = st.selectbox(
                                "Plot Type",
                                ["all", "stacked", "clustered", "mosaic"]
                            )
                            normalize = st.selectbox(
                                "Normalization",
                                [None, "row", "col"],
                                format_func=lambda x: "None (Counts)" if x is None else f"Normalize by {x.capitalize()}"
                            )
                            color_scale = st.selectbox(
                                "Color Scale",
                                ["Cividis", "Viridis", "Plasma", "Inferno", "Jet"],
                                index=0  # Default to colorblind-friendly Cividis
                            )
                            submit_button = st.form_submit_button("Generate Visualizations")

                        if submit_button:
                            if cat_col1 == cat_col2:
                                st.warning("Please select two different categorical columns.")
                            else:
                                with st.spinner("Generating plots..."):
                                    progress_bar = st.progress(0)
                                    try:
                                        result = eda_engine.plot_categorical_vs_categorical(
                                            df=df,
                                            cat_col1=cat_col1,
                                            cat_col2=cat_col2,
                                            target_col=target_col,
                                            plot_type=plot_type,
                                            normalize=normalize,
                                            color_scale=color_scale,
                                            return_figs=True,
                                            max_categories=15
                                        )
                                        if result["figures"]:
                                            st.success(f"Generated {len(result['figures'])} plots.")
                                            for i, fig in enumerate(result["figures"]):
                                                if isinstance(fig, plt.Figure):
                                                    st.pyplot(fig)
                                                else:
                                                    st.plotly_chart(fig, use_container_width=True)
                                                progress_bar.progress((i + 1) / len(result["figures"]))
                                        else:
                                            st.warning("No plots generated. Please check your selections.")

                                        # Display contingency table and chi-square results
                                        with st.expander("Contingency Table & Chi-Square Results", expanded=True):
                                            st.write("**Contingency Table**")
                                            st.dataframe(result["contingency_table"], use_container_width=True)
                                            # Download contingency table as CSV
                                            csv = result["contingency_table"].to_csv().encode('utf-8')
                                            st.download_button(
                                                label="Download Contingency Table as CSV",
                                                data=csv,
                                                file_name=f"contingency_{cat_col1}_vs_{cat_col2}.csv",
                                                mime="text/csv"
                                            )
                                            if result["chi2_result"]:
                                                st.write("**Chi-Square Test Results**")
                                                st.write(f"Statistic: {result['chi2_result']['statistic']:.2f}")
                                                st.write(f"P-value: {result['chi2_result']['p_value']:.3e}")
                                                st.write(f"Degrees of Freedom: {result['chi2_result']['dof']}")
                                                st.write(f"Interpretation: {result['chi2_result']['interpretation']}")
                                    except Exception as e:
                                        st.error(f"Error generating analysis: {str(e)}")

                with multi_inner_tabs[2]:
                    st.subheader("Numeric vs Categorical Relationships")

                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    if target:
                        cat_cols = [col for col in cat_cols if col != target]

                    if not numeric_cols or not cat_cols:
                        st.warning("At least one numeric and one categorical column are required for analysis.")
                    else:
                        with st.form(key="num_vs_cat_form"):
                            st.markdown("### Settings")
                            num_col = st.selectbox(
                                "Select Numeric Column",
                                options=numeric_cols,
                                key="num_col"
                            )
                            cat_col = st.selectbox(
                                "Select Categorical Column",
                                options=cat_cols,
                                key="cat_col"
                            )
                            target_col = st.selectbox(
                                "Target Column (optional, for hue)",
                                [None] + df.columns.tolist(),
                                index=0,
                                key="num_cat_target"
                            )
                            plot_type = st.selectbox(
                                "Plot Type",
                                ["box", "violin", "strip", "swarm"]
                            )
                            color_palette = st.selectbox(
                                "Color Palette",
                                ["Set2", "Pastel1", "Set1", "Dark2", "Paired"],
                                index=0
                            )
                            sample_size = st.slider(
                                "Sample Size (for plotting)",
                                min_value=100,
                                max_value=min(len(df), 5000),
                                value=min(1000, len(df)),
                                step=100
                            )
                            highlight_outliers = st.checkbox("Highlight Outliers", value=False)
                            alpha = st.slider(
                                "Transparency (Alpha)",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.7,
                                step=0.1
                            )
                            submit_button = st.form_submit_button("Generate Visualization")

                        if submit_button:
                            with st.spinner("Generating plot..."):
                                try:
                                    result = eda_engine.plot_numeric_vs_categorical(
                                        df=df,
                                        num_col=num_col,
                                        cat_col=cat_col,
                                        target_col=target_col,
                                        plot_type=plot_type,
                                        color_palette=color_palette,
                                        return_fig=True,
                                        max_categories=15,
                                        fig_size=(10, 6),
                                        alpha=alpha,
                                        highlight_outliers=highlight_outliers,
                                        sample_size=sample_size
                                    )
                                    if result["figure"]:
                                        st.pyplot(result["figure"])
                                    else:
                                        st.warning("No plot generated. Please check your selections.")
                                    if result["message"]:
                                        st.warning(result["message"])
                                except Exception as e:
                                    st.error(f"Error generating plot: {str(e)}") 

                with multi_inner_tabs[3]:
                    st.subheader("AI-Powered Smart Plotting")
                    st.markdown("*Intelligent plot suggestions that adapt to your data selection*")
                    
                    # Initialize the smart plotting engine
                    engine = eda_engine.SmartPlottingEngine()
                    
                    if df is not None and not df.empty:
                        data_analysis = engine.analyze_data(df)
                        
                        # Step 1: Smart Column Selection
                        st.markdown("### Step 1: Select Your Data")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_col = st.selectbox(
                                "Primary Variable (X-axis)",
                                options=[''] + data_analysis['all'],
                                help="Choose your main variable to analyze",
                                key="smart_x_col"
                            )
                        
                        with col2:
                            y_options = [''] + [col for col in data_analysis['all'] if col != x_col]
                            y_col = st.selectbox(
                                "Secondary Variable (Y-axis) - Optional",
                                options=y_options,
                                help="Choose a second variable to explore relationships",
                                key="smart_y_col"
                            )
                        
                        # Step 2: AI Plot Suggestions
                        if x_col:
                            suggestions = engine.suggest_plots(df, x_col, y_col if y_col else None)
                            
                            if suggestions:
                                st.markdown("### Step 2: AI-Recommended Visualizations")
                                st.markdown("*Based on your data types and selection, here are the best plot options:*")
                                
                                # Create dynamic tabs based on suggestions
                                plot_names = [s['plot'] for s in suggestions]
                                plot_tabs = st.tabs([f"{plot.title()}" for plot in plot_names])
                                
                                for idx, (tab, suggestion) in enumerate(zip(plot_tabs, suggestions)):
                                    with tab:
                                        plot_type = suggestion['plot']
                                        
                                        # AI Explanation
                                        st.success(f"**AI Recommendation:** {suggestion['description']}")
                                        st.info(f"**Best for:** {suggestion['best_for']}")
                                        
                                        # Smart Parameter Configuration
                                        with st.form(key=f"smart_plot_{plot_type}_{idx}"):
                                            st.markdown(f"#### Configure Your {plot_type.title()} Plot")
                                            
                                            params = {
                                                'plot_type': plot_type, 
                                                'x_col': x_col, 
                                                'y_col': y_col if y_col else None
                                            }
                                            
                                            # Get relevant parameters for this plot type
                                            relevant_params = engine.get_relevant_params(plot_type)
                                            
                                            # Dynamic parameter rendering based on plot type
                                            param_columns = st.columns(2)
                                            
                                            # Only show relevant parameters
                                            with param_columns[0]:
                                                if 'hue' in relevant_params['optional']:
                                                    hue_options = ['None'] + [col for col in data_analysis['categorical'] 
                                                                            if col not in [x_col, y_col] and col != target]
                                                    if len(hue_options) > 1:
                                                        hue_selection = st.selectbox(
                                                            "Color Groups (Hue)",
                                                            options=hue_options,
                                                            help="Group data by colors",
                                                            key=f"hue_{plot_type}_{idx}"
                                                        )
                                                        params['hue_col'] = hue_selection if hue_selection != 'None' else None
                                                
                                                if 'size' in relevant_params['optional']:
                                                    size_options = ['None'] + [col for col in data_analysis['numeric'] 
                                                                            if col not in [x_col, y_col]]
                                                    if len(size_options) > 1:
                                                        size_selection = st.selectbox(
                                                            "Size Points By",
                                                            options=size_options,
                                                            help="Vary point sizes by numeric values",
                                                            key=f"size_{plot_type}_{idx}"
                                                        )
                                                        params['size_col'] = size_selection if size_selection != 'None' else None
                                            
                                            with param_columns[1]:
                                                if 'style' in relevant_params['optional']:
                                                    style_options = ['None'] + [col for col in data_analysis['categorical'] 
                                                                            if col not in [x_col, y_col, params.get('hue_col')] and col != target]
                                                    if len(style_options) > 1:
                                                        style_selection = st.selectbox(
                                                            "Style Markers By",
                                                            options=style_options,
                                                            help="Use different marker styles",
                                                            key=f"style_{plot_type}_{idx}"
                                                        )
                                                        params['style_col'] = style_selection if style_selection != 'None' else None
                                            
                                            # Plot-specific smart parameters
                                            if plot_type == 'scatter':
                                                st.markdown("##### Scatter Plot Options")
                                                scatter_cols = st.columns(3)
                                                with scatter_cols[0]:
                                                    params['alpha'] = st.slider(
                                                        "Transparency", 0.1, 1.0, 0.7, 0.1,
                                                        help="0.1 = very transparent, 1.0 = solid",
                                                        key=f"alpha_{plot_type}_{idx}"
                                                    )
                                                with scatter_cols[1]:
                                                    params['point_size'] = st.slider(
                                                        "Point Size", 20, 200, 60, 10,
                                                        key=f"point_size_{plot_type}_{idx}"
                                                    )
                                                with scatter_cols[2]:
                                                    params['show_regression'] = st.checkbox(
                                                        "Trend Line", 
                                                        help="Add regression line to show correlation",
                                                        key=f"regression_{plot_type}_{idx}"
                                                    )
                                            
                                            elif plot_type == 'histogram':
                                                st.markdown("##### Histogram Options")
                                                hist_cols = st.columns(2)
                                                with hist_cols[0]:
                                                    params['bins'] = st.slider(
                                                        "Number of Bins", 10, 100, 30, 5,
                                                        help="More bins = more detail",
                                                        key=f"bins_{plot_type}_{idx}"
                                                    )
                                                with hist_cols[1]:
                                                    params['stat'] = st.selectbox(
                                                        "Display Type",
                                                        ['count', 'density', 'probability'],
                                                        help="How to calculate bar heights",
                                                        key=f"stat_{plot_type}_{idx}"
                                                    )
                                            
                                            elif plot_type in ['box', 'violin']:
                                                st.markdown(f"##### {plot_type.title()} Plot Options")
                                                box_cols = st.columns(2)
                                                with box_cols[0]:
                                                    params['orientation'] = st.radio(
                                                        "Orientation", 
                                                        ['vertical', 'horizontal'],
                                                        help="How to orient the plot",
                                                        key=f"orient_{plot_type}_{idx}"
                                                    )
                                                
                                                if plot_type == 'box':
                                                    with box_cols[1]:
                                                        params['show_outliers'] = st.checkbox(
                                                            "Show Outliers", True,
                                                            help="Display data points beyond whiskers",
                                                            key=f"outliers_{plot_type}_{idx}"
                                                        )
                                                elif plot_type == 'violin':
                                                    with box_cols[1]:
                                                        params['inner'] = st.selectbox(
                                                            "Inner Display",
                                                            ['box', 'quart', 'point', 'stick'],
                                                            help="What to show inside violin",
                                                            key=f"inner_{plot_type}_{idx}"
                                                        )
                                            
                                            elif plot_type == 'bar':
                                                st.markdown("##### Bar Chart Options")
                                                params['aggregation'] = st.selectbox(
                                                    "How to Combine Values",
                                                    ['mean', 'sum', 'count', 'median'],
                                                    help="Method for aggregating multiple values",
                                                    key=f"aggregation_{plot_type}_{idx}"
                                                )
                                            
                                            elif plot_type == 'heatmap':
                                                st.markdown("##### Heatmap Options")
                                                value_options = [col for col in data_analysis['numeric'] 
                                                            if col not in [x_col, y_col]]
                                                if value_options:
                                                    heatmap_cols = st.columns(2)
                                                    with heatmap_cols[0]:
                                                        params['value_col'] = st.selectbox(
                                                            "Value to Display",
                                                            value_options,
                                                            help="Which numeric column to show in heatmap",
                                                            key=f"value_{plot_type}_{idx}"
                                                        )
                                                        params['aggregation'] = st.selectbox(
                                                            "Aggregation Method",
                                                            ['mean', 'sum', 'count', 'median'],
                                                            key=f"heatmap_agg_{plot_type}_{idx}"
                                                        )
                                                    with heatmap_cols[1]:
                                                        params['show_annotations'] = st.checkbox(
                                                            "Show Values", True,
                                                            help="Display numbers in each cell",
                                                            key=f"annotations_{plot_type}_{idx}"
                                                        )
                                                        params['colormap'] = st.selectbox(
                                                            "Color Scheme",
                                                            ['viridis', 'plasma', 'coolwarm', 'RdYlBu', 'Blues'],
                                                            key=f"colormap_{plot_type}_{idx}"
                                                        )
                                                else:
                                                    st.warning("Need numeric columns for heatmap values")
                                            
                                            elif plot_type == 'kde':
                                                st.markdown("##### Density Plot Options")
                                                if y_col:
                                                    params['bivariate'] = st.checkbox(
                                                        "2D Density Plot",
                                                        help="Show density for both X and Y together",
                                                        key=f"bivariate_{plot_type}_{idx}"
                                                    )
                                                params['alpha'] = st.slider(
                                                    "Transparency", 0.3, 1.0, 0.7, 0.1,
                                                    key=f"kde_alpha_{plot_type}_{idx}"
                                                )
                                            
                                            elif plot_type == 'jointplot':
                                                st.markdown("##### Joint Plot Options")
                                                params['kind'] = st.selectbox(
                                                    "Plot Style",
                                                    ['scatter', 'reg', 'resid', 'kde', 'hex'],
                                                    help="Type of joint plot to create",
                                                    key=f"joint_kind_{plot_type}_{idx}"
                                                )
                                            
                                            elif plot_type == 'pairplot':
                                                st.markdown("##### Pair Plot Options")
                                                available_numeric = [col for col in data_analysis['numeric'] if col != target]
                                                if len(available_numeric) > 2:
                                                    params['columns'] = st.multiselect(
                                                        "Select Columns to Include",
                                                        available_numeric,
                                                        default=available_numeric[:min(4, len(available_numeric))],
                                                        help="Choose which numeric columns to compare",
                                                        key=f"pairplot_cols_{plot_type}_{idx}"
                                                    )
                                                    params['diag_kind'] = st.selectbox(
                                                        "Diagonal Plots",
                                                        ['hist', 'kde'],
                                                        help="What to show on diagonal",
                                                        key=f"diag_{plot_type}_{idx}"
                                                    )
                                                else:
                                                    st.warning("Need at least 3 numeric columns for pairplot")
                                            
                                            # Advanced options (only if needed)
                                            if len(df) > 1000 or any(col in data_analysis['numeric'] for col in [x_col, y_col]):
                                                with st.expander("Advanced Options", expanded=False):
                                                    adv_cols = st.columns(2)
                                                    
                                                    with adv_cols[0]:
                                                        if len(df) > 1000:
                                                            params['sample_size'] = st.slider(
                                                                "Sample Size (for speed)",
                                                                500, min(10000, len(df)), 
                                                                min(2000, len(df)), 500,
                                                                help="Reduce data size for faster plotting",
                                                                key=f"sample_{plot_type}_{idx}"
                                                            )
                                                        
                                                        if x_col in data_analysis['numeric']:
                                                            params['log_x'] = st.checkbox(
                                                                "Log Scale X-axis",
                                                                help="Use for data spanning many orders of magnitude",
                                                                key=f"log_x_{plot_type}_{idx}"
                                                            )
                                                    
                                                    with adv_cols[1]:
                                                        if y_col and y_col in data_analysis['numeric']:
                                                            params['log_y'] = st.checkbox(
                                                                "Log Scale Y-axis",
                                                                help="Use for data spanning many orders of magnitude",
                                                                key=f"log_y_{plot_type}_{idx}"
                                                            )
                                                        
                                                        # Smart filtering option
                                                        filter_col = st.selectbox(
                                                            "Filter Data By",
                                                            ['None'] + data_analysis['all'],
                                                            help="Optional: filter to subset of data",
                                                            key=f"filter_col_{plot_type}_{idx}"
                                                        )
                                                        
                                                        if filter_col != 'None':
                                                            if filter_col in data_analysis['numeric']:
                                                                filter_range = st.slider(
                                                                    f"Range for {filter_col}",
                                                                    float(df[filter_col].min()),
                                                                    float(df[filter_col].max()),
                                                                    (float(df[filter_col].min()), float(df[filter_col].max())),
                                                                    key=f"filter_range_{plot_type}_{idx}"
                                                                )
                                                                params['filter_condition'] = {
                                                                    'column': filter_col,
                                                                    'range': filter_range
                                                                }
                                                            else:
                                                                unique_values = df[filter_col].unique()[:20]  # Limit options
                                                                filter_value = st.selectbox(
                                                                    f"Value for {filter_col}",
                                                                    unique_values,
                                                                    key=f"filter_value_{plot_type}_{idx}"
                                                                )
                                                                params['filter_condition'] = {
                                                                    'column': filter_col,
                                                                    'value': filter_value
                                                                }
                                            
                                            # Generate button
                                            generate_plot = st.form_submit_button(
                                                f"Create {plot_type.title()} Plot",
                                                help=f"Generate your {plot_type} visualization",
                                                use_container_width=True
                                            )
                                        
                                        # Plot generation
                                        if generate_plot:
                                            with st.spinner(f"Creating your {plot_type} plot..."):
                                                try:
                                                    result = engine.create_smart_plot(
                                                        df, 
                                                        {'plot_type': plot_type, 'params': params}
                                                    )
                                                    
                                                    if result['status'] == 'success' and result['figure']:
                                                        st.success(f"{plot_type.title()} plot created successfully!")
                                                        st.pyplot(result['figure'])
                                                        plt.close()  # Clean up memory
                                                        
                                                        # Show any informational messages
                                                        if result['message']:
                                                            st.info(f"{result['message']}")
                                                        
                                                        # Quick insights (AI-generated)
                                                        with st.expander("Quick Insights", expanded=False):
                                                            if plot_type == 'scatter' and x_col in data_analysis['numeric'] and y_col in data_analysis['numeric']:
                                                                correlation = df[x_col].corr(df[y_col])
                                                                st.write(f"**Correlation between {x_col} and {y_col}:** {correlation:.3f}")
                                                                if abs(correlation) > 0.7:
                                                                    st.write("**Strong relationship detected!**")
                                                                elif abs(correlation) > 0.3:
                                                                    st.write("**Moderate relationship detected**")
                                                                else:
                                                                    st.write("**Weak relationship**")
                                                            
                                                            elif plot_type == 'histogram':
                                                                mean_val = df[x_col].mean()
                                                                median_val = df[x_col].median()
                                                                st.write(f"**{x_col} Statistics:**")
                                                                st.write(f"  • Mean: {mean_val:.2f}")
                                                                st.write(f"  • Median: {median_val:.2f}")
                                                                if abs(mean_val - median_val) / median_val > 0.1:
                                                                    st.write("**Distribution appears skewed**")
                                                                else:
                                                                    st.write("**Distribution appears symmetric**")
                                                    
                                                    else:
                                                        st.error(f"{result['message']}")
                                                
                                                except Exception as e:
                                                    st.error(f"Unexpected error: {str(e)}")
                                                    st.info("Try adjusting your parameters or selecting different columns.")
                            
                            else:
                                st.warning("No suitable plots found for your selection. Try choosing different columns.")
                        
                        else:
                            # Welcome screen with data overview
                            st.markdown("### Welcome to Smart Plotting!")
                            st.markdown("**Get started by selecting a column above**")
                            
                            # Data overview cards
                            st.markdown("### Your Data at a Glance")
                            overview_cols = st.columns(4)
                            
                            with overview_cols[0]:
                                st.metric("Total Rows", f"{len(df):,}")
                            with overview_cols[1]:
                                st.metric("Numeric Cols", len(data_analysis['numeric']))
                            with overview_cols[2]:
                                st.metric("Category Cols", len(data_analysis['categorical']))
                            with overview_cols[3]:
                                st.metric("Date Cols", len(data_analysis['datetime']))
                            
                            # Column previews
                            if data_analysis['numeric']:
                                with st.expander("Numeric Columns (good for scatter, histogram, box plots)", expanded=True):
                                    st.write("• " + "\n• ".join(data_analysis['numeric']))
                            
                            if data_analysis['categorical']:
                                with st.expander("Categorical Columns (good for grouping, colors, count plots)", expanded=True):
                                    st.write("• " + "\n• ".join(data_analysis['categorical']))
                            
                            # Quick suggestions
                            st.markdown("### Quick Start Suggestions")
                            suggestions_cols = st.columns(2)
                            
                            with suggestions_cols[0]:
                                st.info("**For Beginners:** Start with a single numeric column for histograms or a categorical column for count plots")
                            
                            with suggestions_cols[1]:
                                st.info("**For Exploration:** Pick two numeric columns to discover relationships with scatter plots")
                    
                    else:
                        st.error("No data available. Please upload a dataset first.") 

                with multi_inner_tabs[4]:
                    st.markdown("""
                    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
                        <h2 style='color: white; text-align: center; margin: 0;'>
                            AI-Powered Advanced Analysis Engine
                        </h2>
                        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
                            Discover hidden patterns, structural relationships, and complex interactions in your data
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Initialize the analysis engine
                    if 'analysis_engine' not in st.session_state:
                        st.session_state.analysis_engine = eda_engine.AdvancedAnalysisEngine()
                    
                    analysis_engine = st.session_state.analysis_engine
                    
                    # Main control panel
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            target_column = st.selectbox(
                                "Select Target Variable (Optional)",
                                options=["None"] + list(df.columns),
                                help="Choose your target variable for supervised analysis"
                            )
                            target_column = None if target_column == "None" else target_column
                        
                        with col2:
                            analysis_scope = st.multiselect(
                                "Analysis Components",
                                options=[
                                    "Data Structure Analysis",
                                    "Missing Data Patterns", 
                                    "PCA Dimensionality",
                                    "UMAP Manifold",
                                    "Mutual Information",
                                    "Feature Interactions",
                                    "Advanced Correlations"
                                ],
                                default=["Data Structure Analysis", "Missing Data Patterns", "PCA Dimensionality"],
                                help="Select which AI analyses to perform"
                            )
                        
                        with col3:
                            st.markdown("### Quick Actions")
                            run_full_analysis = st.button(
                                "Run Full AI Analysis",
                                type="primary",
                                help="Perform comprehensive automated analysis"
                            )
                            
                            generate_report = st.button(
                                "Generate Report",
                                help="Create executive summary"
                            )
                    
                    # Advanced Options Expander
                    with st.expander("Advanced Configuration", expanded=False):
                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        
                        with adv_col1:
                            st.subheader("Processing Options")
                            max_features = st.slider("Max Features to Analyze", 5, 50, 20)
                            scaling_method = st.selectbox("Scaling Method", ["standard", "minmax", "robust"])
                            
                        with adv_col2:
                            st.subheader("Visualization Options")
                            show_3d_plots = st.checkbox("Enable 3D Visualizations", True)
                            show_heatmaps = st.checkbox("Show Correlation Heatmaps", True)
                            color_by_target = st.checkbox("Color by Target", True if target_column else False)
                            
                        with adv_col3:
                            st.subheader("Algorithm Parameters")
                            pca_components = st.slider("PCA Components", 2, 10, 5)
                            umap_neighbors = st.slider("UMAP Neighbors", 5, 50, 15)
                            correlation_threshold = st.slider("Correlation Threshold", 0.1, 0.9, 0.5)
                    
                    # Analysis Execution
                    if run_full_analysis or any(analysis_scope):
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        analysis_results = {}
                        total_analyses = len(analysis_scope)
                        
                        # 1. Data Structure Analysis
                        if "Data Structure Analysis" in analysis_scope:
                            status_text.text("Analyzing data structure and generating AI recommendations...")
                            progress_bar.progress(1/total_analyses if total_analyses > 0 else 0)
                            
                            try:
                                structure_analysis = analysis_engine.analyze_data_structure(df, target_column)
                                analysis_results['structure'] = structure_analysis
                                
                                # Display results
                                st.markdown("---")
                                st.markdown("## Data Structure Analysis")
                                
                                # Basic info cards
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Features", structure_analysis['basic_info']['n_features'])
                                with col2:
                                    st.metric("Numeric Features", len(structure_analysis['data_types']['numeric']))
                                with col3:
                                    st.metric("Categorical Features", len(structure_analysis['data_types']['categorical']))
                                with col4:
                                    st.metric("Complexity Score", f"{structure_analysis['complexity_score']:.1f}/100")
                                
                                # AI Recommendations
                                if structure_analysis['ai_recommendations']:
                                    st.markdown("### 🤖 AI Recommendations")
                                    for i, rec in enumerate(structure_analysis['ai_recommendations']):
                                        priority_color = {
                                            'high': '🔴',
                                            'medium': '🟡', 
                                            'low': '🟢'
                                        }.get(rec['priority'], '🔵')
                                        
                                        with st.container():
                                            st.markdown(f"""
                                            <div style='border-left: 4px solid #{"dc3545" if rec["priority"]=="high" else "#ffc107" if rec["priority"]=="medium" else "#28a745"}; 
                                                        padding: 1rem; margin: 0.5rem 0; background-color: #f8f9fa; border-radius: 5px;'>
                                                <strong>{priority_color} {rec['title']}</strong><br>
                                                <small style='color: #6c757d;'>{rec['description']}</small><br>
                                                <em>Action: {rec['action']}</em>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                            except Exception as e:
                                st.error(f"Error in structure analysis: {str(e)}")
                        
                        # 2. Missing Data Analysis
                        if "Missing Data Patterns" in analysis_scope:
                            status_text.text("Analyzing missing data patterns...")
                            progress_bar.progress(2/total_analyses if total_analyses > 1 else 1)
                            
                            try:
                                missing_analysis = analysis_engine.create_missingness_analysis(
                                    df, 
                                    advanced_options={
                                        'show_heatmap': show_heatmaps,
                                        'show_correlation': True
                                    }
                                )
                                analysis_results['missing'] = missing_analysis
                                
                                if missing_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## Missing Data Intelligence")
                                    
                                    # Missing data metrics
                                    if 'insights' in missing_analysis:
                                        insights = missing_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Features with Missing Data", 
                                                insights['summary_stats']['features_with_missing']
                                            )
                                        with col2:
                                            st.metric(
                                                "Average Missing %", 
                                                f"{insights['summary_stats']['average_missing_pct']:.1f}%"
                                            )
                                        with col3:
                                            st.metric(
                                                "Total Missing Cells", 
                                                f"{insights['summary_stats']['total_missing_cells']:,}"
                                            )
                                    
                                    # Display figures
                                    fig_col1, fig_col2 = st.columns(2)
                                    
                                    if 'heatmap' in missing_analysis['figures']:
                                        with fig_col1:
                                            st.pyplot(missing_analysis['figures']['heatmap'])
                                    
                                    if 'bar_chart' in missing_analysis['figures']:
                                        with fig_col2:
                                            st.pyplot(missing_analysis['figures']['bar_chart'])
                                            
                                else:
                                    st.info(missing_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in missing data analysis: {str(e)}")
                        
                        # 3. PCA Analysis
                        if "PCA Dimensionality" in analysis_scope:
                            status_text.text("Performing Principal Component Analysis...")
                            progress_bar.progress(3/total_analyses if total_analyses > 2 else 1)
                            
                            try:
                                pca_analysis = analysis_engine.create_pca_analysis(
                                    df, 
                                    target=target_column,
                                    scaling_method=scaling_method,
                                    n_components=pca_components,
                                    advanced_options={'show_loadings': True}
                                )
                                analysis_results['pca'] = pca_analysis
                                
                                if pca_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## Principal Component Analysis")
                                    
                                    # PCA metrics
                                    if 'insights' in pca_analysis:
                                        insights = pca_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Components for 80% Variance", 
                                                insights['components_for_80_percent']
                                            )
                                        with col2:
                                            st.metric(
                                                "Components for 95% Variance", 
                                                insights['components_for_95_percent']
                                            )
                                        with col3:
                                            st.metric(
                                                "Dimensionality Reduction", 
                                                f"{insights['dimensionality_reduction']['original_features']} → {insights['dimensionality_reduction']['recommended_components']}"
                                            )
                                    
                                    # Display PCA figures
                                    if 'variance_explained' in pca_analysis['figures']:
                                        st.pyplot(pca_analysis['figures']['variance_explained'])
                                    
                                    fig_col1, fig_col2 = st.columns(2)
                                    if 'pca_scatter' in pca_analysis['figures']:
                                        with fig_col1:
                                            st.pyplot(pca_analysis['figures']['pca_scatter'])
                                    
                                    if 'loadings' in pca_analysis['figures']:
                                        with fig_col2:
                                            st.pyplot(pca_analysis['figures']['loadings'])
                                            
                                else:
                                    st.warning(pca_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in PCA analysis: {str(e)}")
                        
                        # 4. UMAP Analysis
                        if "UMAP Manifold" in analysis_scope:
                            status_text.text("Creating UMAP manifold projection...")
                            progress_bar.progress(4/total_analyses if total_analyses > 3 else 1)
                            
                            try:
                                umap_analysis = analysis_engine.create_umap_analysis(
                                    df,
                                    target=target_column,
                                    n_neighbors=umap_neighbors,
                                    min_dist=0.1,
                                    advanced_options={'show_clusters': True}
                                )
                                analysis_results['umap'] = umap_analysis
                                
                                if umap_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## UMAP Manifold Learning")
                                    
                                    # UMAP metrics
                                    if 'insights' in umap_analysis:
                                        insights = umap_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Samples Used", insights['n_samples_used'])
                                        with col2:
                                            st.metric("Features Used", insights['n_features_used'])
                                        with col3:
                                            if 'cluster_analysis' in insights:
                                                st.metric(
                                                    "Estimated Clusters", 
                                                    insights['cluster_analysis']['estimated_clusters']
                                                )
                                    
                                    # Display UMAP figure
                                    if 'umap_2d' in umap_analysis['figures']:
                                        st.pyplot(umap_analysis['figures']['umap_2d'])
                                        
                                else:
                                    st.warning(umap_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in UMAP analysis: {str(e)}")
                        
                        # 5. Mutual Information Analysis
                        if "Mutual Information" in analysis_scope and target_column:
                            status_text.text("Computing mutual information relationships...")
                            progress_bar.progress(5/total_analyses if total_analyses > 4 else 1)
                            
                            try:
                                mi_analysis = analysis_engine.create_mutual_info_analysis(
                                    df,
                                    target=target_column,
                                    feature_types='all',
                                    max_features=max_features
                                )
                                analysis_results['mutual_info'] = mi_analysis
                                
                                if mi_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## Mutual Information Analysis")
                                    
                                    # MI metrics
                                    if 'insights' in mi_analysis:
                                        insights = mi_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Features Analyzed", insights['features_analyzed'])
                                        with col2:
                                            st.metric("Average MI Score", f"{insights['average_mi_score']:.3f}")
                                        with col3:
                                            st.metric("Max MI Score", f"{insights['max_mi_score']:.3f}")
                                        
                                        # Top features
                                        if insights['top_features']:
                                            st.markdown("### Top Predictive Features")
                                            top_features_df = pd.DataFrame(insights['top_features'])
                                            st.dataframe(
                                                top_features_df[['Feature', 'Mutual_Information', 'Type']].round(3),
                                                use_container_width=True
                                            )
                                    
                                    # Display MI figure
                                    if 'mutual_info' in mi_analysis['figures']:
                                        st.pyplot(mi_analysis['figures']['mutual_info'])
                                        
                                else:
                                    st.warning(mi_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in mutual information analysis: {str(e)}")
                        
                        # 6. Feature Interactions
                        if "Feature Interactions" in analysis_scope and target_column:
                            status_text.text("Analyzing feature interactions...")
                            progress_bar.progress(6/total_analyses if total_analyses > 5 else 1)
                            
                            try:
                                interaction_analysis = analysis_engine.create_feature_interaction_analysis(
                                    df,
                                    target=target_column,
                                    interaction_method='correlation',
                                    max_interactions=15,
                                    advanced_options={
                                        'show_comparison': True,
                                        'show_3d_surface': show_3d_plots
                                    }
                                )
                                analysis_results['interactions'] = interaction_analysis
                                
                                if interaction_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## Feature Interaction Intelligence")
                                    
                                    # Interaction metrics
                                    if 'insights' in interaction_analysis:
                                        insights = interaction_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Interactions Analyzed", 
                                                insights['total_interactions_analyzed']
                                            )
                                        with col2:
                                            st.metric(
                                                "Beneficial Interactions", 
                                                insights['beneficial_interactions']
                                            )
                                        with col3:
                                            st.metric(
                                                "Avg Interaction Strength", 
                                                f"{insights['average_interaction_strength']:.3f}"
                                            )
                                        
                                        # Top interaction
                                        if 'top_interaction' in insights:
                                            st.info(f"**Strongest Interaction**: {insights['top_interaction']['features']} "
                                                f"(Strength: {insights['top_interaction']['strength']:.3f})")
                                    
                                    # Display interaction figures
                                    if 'interaction_strength' in interaction_analysis['figures']:
                                        st.pyplot(interaction_analysis['figures']['interaction_strength'])
                                    
                                    if 'comparison' in interaction_analysis['figures']:
                                        st.pyplot(interaction_analysis['figures']['comparison'])
                                    
                                    if show_3d_plots and '3d_surface' in interaction_analysis['figures']:
                                        st.pyplot(interaction_analysis['figures']['3d_surface'])
                                        
                                else:
                                    st.warning(interaction_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in interaction analysis: {str(e)}")
                        
                        # 7. Advanced Correlations
                        if "Advanced Correlations" in analysis_scope:
                            status_text.text("Computing advanced correlation patterns...")
                            progress_bar.progress(1.0)
                            
                            try:
                                corr_analysis = analysis_engine.create_advanced_correlation_analysis(
                                    df,
                                    correlation_method='pearson',
                                    cluster_method='ward',
                                    advanced_options={
                                        'show_clustered_heatmap': show_heatmaps,
                                        'show_network': True,
                                        'correlation_threshold': correlation_threshold
                                    }
                                )
                                analysis_results['correlations'] = corr_analysis
                                
                                if corr_analysis['status'] == 'success':
                                    st.markdown("---")
                                    st.markdown("## Advanced Correlation Intelligence")
                                    
                                    # Correlation metrics
                                    if 'insights' in corr_analysis:
                                        insights = corr_analysis['insights']
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Mean Abs Correlation", 
                                                f"{insights['correlation_summary']['mean_absolute_correlation']:.3f}"
                                            )
                                        with col2:
                                            st.metric(
                                                "High Correlation Pairs", 
                                                insights['correlation_summary']['high_correlation_pairs']
                                            )
                                        with col3:
                                            st.metric(
                                                "Multicollinear Features", 
                                                insights['multicollinearity_analysis']['features_with_high_correlation']
                                            )
                                    
                                    # Display correlation figures
                                    fig_col1, fig_col2 = st.columns(2)
                                    
                                    if 'clustered_heatmap' in corr_analysis['figures']:
                                        with fig_col1:
                                            st.pyplot(corr_analysis['figures']['clustered_heatmap'])
                                    
                                    if 'network' in corr_analysis['figures']:
                                        with fig_col2:
                                            st.pyplot(corr_analysis['figures']['network'])
                                    
                                    if 'distribution' in corr_analysis['figures']:
                                        st.pyplot(corr_analysis['figures']['distribution'])
                                        
                                else:
                                    st.warning(corr_analysis['message'])
                                    
                            except Exception as e:
                                st.error(f"Error in correlation analysis: {str(e)}")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Success message
                        st.success(f"Advanced AI analysis completed! Analyzed {len(analysis_results)} components.")
                    
                    # Generate Comprehensive Report
                    if generate_report:
                        st.markdown("---")
                        st.markdown("## AI Executive Report")
                        
                        with st.spinner("Generating comprehensive AI report..."):
                            try:
                                report = analysis_engine.create_comprehensive_report(
                                    df, 
                                    target=target_column,
                                    analyses_to_include=[
                                        'structure', 'missingness', 'pca', 
                                        'mutual_info', 'interactions'
                                    ]
                                )
                                
                                # Executive Summary
                                if 'executive_summary' in report:
                                    summary = report['executive_summary']
                                    
                                    st.markdown("### Executive Summary")
                                    
                                    # Overview metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Dataset Size", f"{summary['dataset_overview']['rows']:,} rows")
                                    with col2:
                                        st.metric("Features", summary['dataset_overview']['features'])
                                    with col3:
                                        st.metric("Memory Usage", f"{summary['dataset_overview']['memory_mb']:.1f} MB")
                                    with col4:
                                        data_quality_color = {
                                            'Ready for modeling': 'green',
                                            'Minor preprocessing needed': 'orange',
                                            'Significant preprocessing required': 'red'
                                        }.get(summary['data_readiness'], 'blue')
                                        
                                        st.markdown(f"""
                                        <div style='text-align: center; padding: 0.5rem; border-radius: 5px; 
                                                    background-color: {data_quality_color}; color: white; font-weight: bold;'>
                                            {summary['data_readiness']}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Key findings
                                    if summary['key_findings']:
                                        st.markdown("### Key Findings")
                                        for finding in summary['key_findings']:
                                            st.markdown(f"• {finding}")
                                
                                # AI Recommendations
                                if report['ai_recommendations']:
                                    st.markdown("### AI Recommendations")
                                    for i, rec in enumerate(report['ai_recommendations'][:5], 1):
                                        st.markdown(f"{i}. {rec}")
                                
                                # Action Items
                                if report['action_items']:
                                    st.markdown("### Prioritized Action Items")
                                    action_df = pd.DataFrame(report['action_items'])
                                    st.dataframe(action_df, use_container_width=True)
                                
                                # Data Quality Score
                                if 'data_quality_score' in report:
                                    st.markdown("### Data Quality Assessment")
                                    quality_score = report['data_quality_score']
                                    
                                    # Create quality gauge
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.metric("Overall Quality Score", f"{quality_score:.1f}/100")
                                    
                                    with col2:
                                        # Simple progress bar visualization
                                        st.progress(quality_score / 100)
                                        
                                        if quality_score >= 80:
                                            st.success("Excellent data quality! 🌟")
                                        elif quality_score >= 60:
                                            st.info("Good data quality with room for improvement 👍")
                                        else:
                                            st.warning("Data quality needs significant improvement ⚠️")
                                
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
                    
                    # Footer with tips
                    st.markdown("---")
                    with st.expander("Pro Tips for Advanced Analysis", expanded=False):
                        st.markdown("""
                        **Target Variable Selection:**
                        - Choose your prediction target for supervised analysis insights
                        - Leave empty for unsupervised exploration
                        
                        **Analysis Component Guide:**
                        - **Data Structure**: Overall data health and complexity assessment
                        - **Missing Data**: Understand missingness patterns and impact
                        - **PCA**: Identify which features capture most variance
                        - **UMAP**: Discover non-linear patterns and clusters
                        - **Mutual Information**: Find features most predictive of target
                        - **Feature Interactions**: Discover synergistic feature combinations
                        - **Advanced Correlations**: Detect multicollinearity and feature groups
                        
                        **Performance Tips:**
                        - Limit max features for large datasets
                        - Use standard scaling for mixed-range features
                        - Enable 3D plots only for key insights (computationally intensive)
                        
                        **AI Recommendations:**
                        - Red alerts require immediate attention
                        - Yellow suggestions can improve model performance
                        - Green insights are optimization opportunities
                        """) 

            with adv_tab:
                adv_tab, report_tab = st.tabs(['Advance Insights','Report'])
                with adv_tab:
                    st.header("AI-Powered Advanced Data Analysis")
                    st.markdown("**Comprehensive automated insights generation with advanced statistical analysis**")
                    
                    # Initialize the insight generator
                    @st.cache_resource
                    def get_insight_generator():
                        return eda_engine.RobustAutomatedInsightGenerator()
                    
                    insight_generator = get_insight_generator()
                    
                    if uploaded_file is not None and df is not None:
                        # Configuration Section
                        st.subheader("Analysis Configuration")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Target variable selection
                            target_options = ['None'] + list(df.columns)
                            selected_target = st.selectbox(
                                "Select Target Variable (Optional)",
                                target_options,
                                help="Choose the target variable for supervised learning analysis"
                            )
                            
                            target_column = selected_target if selected_target != 'None' else None
                        
                        with col2:
                            # Analysis options
                            st.markdown("**Analysis Options:**")
                            include_business_insights = st.checkbox("Include Business Insights", value=True)
                            include_ai_recommendations = st.checkbox("Include AI Recommendations", value=True)
                            show_detailed_warnings = st.checkbox("Show Detailed Warnings", value=False)
                        
                        # Analysis execution
                        st.subheader("Generate Advanced Analysis")
                        
                        if st.button("Run Comprehensive Analysis", type="primary", use_container_width=True):
                            with st.spinner("AI is analyzing your data... This may take a moment for large datasets."):
                                try:
                                    # Generate insights
                                    insights = insight_generator.generate_insights(df, target_column)
                                    
                                    # Check for errors
                                    if insights.get('error'):
                                        st.error(f"Analysis failed: {insights.get('message', 'Unknown error')}")
                                        if 'recommendations' in insights:
                                            st.info("**Recommendations:**")
                                            for rec in insights['recommendations']:
                                                st.write(f"• {rec}")
                                    else:
                                        # Success - Display results
                                        st.success("Analysis completed successfully!")
                                        
                                        # Store insights in session state
                                        st.session_state['advanced_insights'] = insights
                                        st.session_state['summary_report'] = insight_generator.generate_summary_report()
                                        
                                except Exception as e:
                                    st.error(f"Critical error during analysis: {str(e)}")
                                    st.info("Please check your data format and try again.")
                        
                        # Display results if available
                        if 'advanced_insights' in st.session_state:
                            insights = st.session_state['advanced_insights']
                            
                            # Summary Report
                            st.subheader("Executive Summary")
                            with st.expander("Comprehensive Analysis Report", expanded=True):
                                st.text(st.session_state.get('summary_report', 'Report not available'))
                            
                            # Key Insights
                            st.subheader("Key Insights")
                            key_insights = insights.get('key_insights', [])
                            if key_insights:
                                for i, insight in enumerate(key_insights, 1):
                                    if isinstance(insight, dict):
                                        priority = insight.get('priority', 0)
                                        action_needed = insight.get('action_needed', False)
                                        
                                        # Color coding based on priority and action needed
                                        if action_needed and priority >= 8:
                                            st.error(f"**{i}.** {insight.get('insight', 'N/A')}")
                                        elif action_needed:
                                            st.warning(f"**{i}.** {insight.get('insight', 'N/A')}")
                                        else:
                                            st.info(f"**{i}.** {insight.get('insight', 'N/A')}")
                            else:
                                st.write("No key insights generated.")
                            
                            # Detailed Analysis Sections
                            tabs = st.tabs([
                                "Data Composition", 
                                "Target Analysis", 
                                "Feature Patterns", 
                                "Relationships",
                                "Data Quality", 
                                "AI Recommendations",
                                "Business Insights"
                            ])
                            
                            # Data Composition Tab
                            with tabs[0]:
                                st.subheader("Data Composition Analysis")
                                
                                data_composition = insights.get('data_composition', {})
                                if data_composition and not data_composition.get('error'):
                                    
                                    # Dataset Shape
                                    shape_info = data_composition.get('shape', {})
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Total Rows", f"{shape_info.get('rows', 0):,}")
                                    with col2:
                                        st.metric("Total Columns", shape_info.get('columns', 0))
                                    with col3:
                                        st.metric("Data Density", f"{data_composition.get('data_density', 0)}%")
                                    with col4:
                                        memory_info = data_composition.get('memory_usage', {})
                                        st.metric("Memory Usage", f"{memory_info.get('total_memory_mb', 0):.1f} MB")
                                    
                                    # Feature Types
                                    st.subheader("Feature Type Distribution")
                                    feature_types = data_composition.get('feature_types', {})
                                    
                                    type_data = {
                                        'Type': ['Numerical', 'Categorical', 'DateTime', 'Boolean'],
                                        'Count': [
                                            feature_types.get('numerical', 0),
                                            feature_types.get('categorical', 0),
                                            feature_types.get('datetime', 0),
                                            feature_types.get('boolean', 0)
                                        ]
                                    }
                                    
                                    # Create a simple bar chart
                                    type_df = pd.DataFrame(type_data)
                                    st.bar_chart(type_df.set_index('Type'))
                                    
                                    # Feature Lists
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Numerical Features")
                                        numerical_features = feature_types.get('numerical_features', [])
                                        if numerical_features:
                                            for feature in numerical_features:
                                                st.write(f"• {feature}")
                                        else:
                                            st.write("No numerical features found.")
                                        
                                        st.subheader("DateTime Features")
                                        datetime_features = feature_types.get('datetime_features', [])
                                        if datetime_features:
                                            for feature in datetime_features:
                                                st.write(f"• {feature}")
                                        else:
                                            st.write("No datetime features found.")
                                    
                                    with col2:
                                        st.subheader("Categorical Features")
                                        categorical_features = feature_types.get('categorical_features', [])
                                        if categorical_features:
                                            for feature in categorical_features:
                                                st.write(f"• {feature}")
                                        else:
                                            st.write("No categorical features found.")
                                        
                                        st.subheader("Boolean Features")
                                        boolean_features = feature_types.get('boolean_features', [])
                                        if boolean_features:
                                            for feature in boolean_features:
                                                st.write(f"• {feature}")
                                        else:
                                            st.write("No boolean features found.")
                                    
                                    # Missing Data Analysis
                                    st.subheader("Missing Data Analysis")
                                    missing_data = data_composition.get('missing_data', {})
                                    
                                    if missing_data and not missing_data.get('error'):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Total Missing Values", f"{missing_data.get('total_missing', 0):,}")
                                        with col2:
                                            st.metric("Missing Percentage", f"{missing_data.get('missing_percentage', 0):.2f}%")
                                        with col3:
                                            high_missing = missing_data.get('high_missing_columns', [])
                                            st.metric("High Missing Columns", len(high_missing))
                                        
                                        # Missing data by column
                                        by_column = missing_data.get('by_column', {})
                                        if by_column:
                                            st.subheader("Missing Data by Column")
                                            missing_df = pd.DataFrame([
                                                {'Column': col, 'Missing Count': info.get('count', 0), 'Missing %': info.get('percentage', 0)}
                                                for col, info in by_column.items() if info.get('percentage', 0) > 0
                                            ])
                                            
                                            if not missing_df.empty:
                                                st.dataframe(missing_df.sort_values('Missing %', ascending=False), use_container_width=True)
                                            else:
                                                st.info("No missing values detected in any column!")
                                
                                else:
                                    st.error("Data composition analysis failed or unavailable.")
                            
                            # Target Analysis Tab
                            with tabs[1]:
                                st.subheader("Target Variable Analysis")
                                
                                target_insights = insights.get('target_insights', {})
                                
                                if target_insights.get('message'):
                                    st.info(target_insights['message'])
                                elif target_insights.get('error'):
                                    st.error(f"{target_insights['error']}")
                                else:
                                    # Basic target info
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Target Column", target_insights.get('column_name', 'N/A'))
                                    with col2:
                                        st.metric("Data Type", target_insights.get('data_type', 'N/A'))
                                    with col3:
                                        st.metric("Missing %", f"{target_insights.get('missing_percentage', 0):.2f}%")
                                    with col4:
                                        st.metric("Unique Values", target_insights.get('unique_values', 0))
                                    
                                    # Task type and analysis
                                    task_type = target_insights.get('task_type', 'unknown')
                                    st.subheader(f"Problem Type: {task_type.title()}")
                                    
                                    if task_type == 'classification':
                                        class_balance = target_insights.get('class_balance', {})
                                        
                                        if class_balance and not class_balance.get('error'):
                                            # Class balance metrics
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric("Number of Classes", class_balance.get('n_classes', 0))
                                            with col2:
                                                st.metric("Imbalance Ratio", f"{class_balance.get('imbalance_ratio', 0):.2f}")
                                            with col3:
                                                severity = class_balance.get('severity', 'unknown')
                                                if severity == 'high':
                                                    st.error(f"Balance Status: {class_balance.get('status', 'N/A')}")
                                                elif severity == 'medium':
                                                    st.warning(f"Balance Status: {class_balance.get('status', 'N/A')}")
                                                else:
                                                    st.success(f"Balance Status: {class_balance.get('status', 'N/A')}")
                                            
                                            # Class details
                                            st.subheader("Class Distribution")
                                            class_details = class_balance.get('class_details', [])
                                            if class_details:
                                                class_df = pd.DataFrame(class_details)
                                                st.dataframe(class_df, use_container_width=True)
                                        
                                        # Recommendations
                                        recommendations = target_insights.get('recommendations', [])
                                        if recommendations:
                                            st.subheader("Classification Recommendations")
                                            for rec in recommendations:
                                                if rec.startswith('⚠️') or rec.startswith('🚨'):
                                                    st.warning(rec)
                                                elif rec.startswith('✅'):
                                                    st.success(rec)
                                                else:
                                                    st.info(rec)
                                    
                                    elif task_type == 'regression':
                                        # Statistics
                                        stats = target_insights.get('statistics', {})
                                        if stats:
                                            st.subheader("Target Variable Statistics")
                                            
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Mean", f"{stats.get('mean', 0):.4f}")
                                            with col2:
                                                st.metric("Std Dev", f"{stats.get('std', 0):.4f}")
                                            with col3:
                                                st.metric("Min", f"{stats.get('min', 0):.4f}")
                                            with col4:
                                                st.metric("Max", f"{stats.get('max', 0):.4f}")
                                            
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Median", f"{stats.get('median', 0):.4f}")
                                            with col2:
                                                st.metric("Skewness", f"{stats.get('skewness', 0):.4f}")
                                            with col3:
                                                st.metric("Kurtosis", f"{stats.get('kurtosis', 0):.4f}")
                                            with col4:
                                                st.metric("Range", f"{stats.get('range', 0):.4f}")
                                        
                                        # Distribution analysis
                                        distribution = target_insights.get('distribution', {})
                                        if distribution:
                                            st.subheader("Distribution Analysis")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write(f"**Shape:** {distribution.get('shape', 'N/A')}")
                                                normality = distribution.get('normality', {})
                                                if normality.get('is_normal'):
                                                    st.success("Normally distributed")
                                                else:
                                                    st.warning("Not normally distributed")
                                            
                                            with col2:
                                                if distribution.get('transformation_needed'):
                                                    st.warning("Transformation recommended")
                                                else:
                                                    st.success("No transformation needed")
                                        
                                        # Recommendations
                                        recommendations = target_insights.get('recommendations', [])
                                        if recommendations:
                                            st.subheader("Regression Recommendations")
                                            for rec in recommendations:
                                                if rec.startswith('⚠️') or rec.startswith('📈'):
                                                    st.warning(rec)
                                                elif rec.startswith('✅'):
                                                    st.success(rec)
                                                else:
                                                    st.info(rec)
                                    
                                    # Top correlated features
                                    top_correlated = target_insights.get('top_correlated_features', [])
                                    if top_correlated:
                                        st.subheader("Top Correlated Features")
                                        corr_df = pd.DataFrame(top_correlated)
                                        if not corr_df.empty:
                                            st.dataframe(corr_df, use_container_width=True)
                            
                            # Feature Patterns Tab
                            with tabs[2]:
                                st.subheader("Feature Pattern Analysis")
                                
                                feature_patterns = insights.get('feature_patterns', {})
                                
                                if feature_patterns.get('error'):
                                    st.error(f"{feature_patterns['error']}")
                                else:
                                    # Numerical patterns
                                    numerical_patterns = feature_patterns.get('numerical_patterns', {})
                                    if numerical_patterns:
                                        st.subheader("Numerical Feature Patterns")
                                        
                                        # Create summary table
                                        num_summary = []
                                        for col, pattern in numerical_patterns.items():
                                            if isinstance(pattern, dict) and not pattern.get('error'):
                                                issues = pattern.get('potential_issues', [])
                                                num_summary.append({
                                                    'Feature': col,
                                                    'Mean': f"{pattern.get('mean', 0):.4f}",
                                                    'Std Dev': f"{pattern.get('std', 0):.4f}",
                                                    'Skewness': f"{pattern.get('skewness', 0):.4f}",
                                                    'Outliers %': f"{pattern.get('outliers', {}).get('iqr_method', {}).get('percentage', 0):.1f}%",
                                                    'Issues': ', '.join(issues) if issues else 'None'
                                                })
                                        
                                        if num_summary:
                                            num_df = pd.DataFrame(num_summary)
                                            st.dataframe(num_df, use_container_width=True)
                                        
                                        # Detailed view for problematic features
                                        problematic_features = [
                                            col for col, pattern in numerical_patterns.items()
                                            if isinstance(pattern, dict) and pattern.get('potential_issues')
                                        ]
                                        
                                        if problematic_features:
                                            st.subheader("Features Requiring Attention")
                                            selected_feature = st.selectbox("Select feature for detailed analysis:", problematic_features)
                                            
                                            if selected_feature:
                                                pattern = numerical_patterns[selected_feature]
                                                issues = pattern.get('potential_issues', [])
                                                
                                                for issue in issues:
                                                    if issue == 'highly_skewed':
                                                        st.warning(f"{selected_feature} is highly skewed (skewness: {pattern.get('skewness', 0):.3f})")
                                                    elif issue == 'many_outliers':
                                                        outlier_pct = pattern.get('outliers', {}).get('iqr_method', {}).get('percentage', 0)
                                                        st.warning(f"{selected_feature} has many outliers ({outlier_pct:.1f}%)")
                                                    elif issue == 'low_variance':
                                                        st.warning(f"{selected_feature} has low variance")
                                                    elif issue == 'mostly_zeros':
                                                        zero_pct = pattern.get('zero_percentage', 0)
                                                        st.warning(f"{selected_feature} is mostly zeros ({zero_pct:.1f}%)")
                                    
                                    # Categorical patterns
                                    categorical_patterns = feature_patterns.get('categorical_patterns', {})
                                    if categorical_patterns:
                                        st.subheader("Categorical Feature Patterns")
                                        
                                        cat_summary = []
                                        for col, pattern in categorical_patterns.items():
                                            if isinstance(pattern, dict) and not pattern.get('error'):
                                                issues = pattern.get('potential_issues', [])
                                                cat_summary.append({
                                                    'Feature': col,
                                                    'Unique Values': pattern.get('unique_values', 0),
                                                    'Unique Ratio': f"{pattern.get('unique_ratio', 0):.4f}",
                                                    'Dominant Value': pattern.get('dominant_value', 'N/A')[:20] + '...' if len(str(pattern.get('dominant_value', ''))) > 20 else pattern.get('dominant_value', 'N/A'),
                                                    'Dominant %': f"{pattern.get('dominant_percentage', 0):.1f}%",
                                                    'Issues': ', '.join(issues) if issues else 'None'
                                                })
                                        
                                        if cat_summary:
                                            cat_df = pd.DataFrame(cat_summary)
                                            st.dataframe(cat_df, use_container_width=True)
                                    
                                    # Feature correlations
                                    correlations = feature_patterns.get('feature_correlations', [])
                                    if correlations:
                                        st.subheader("Strong Feature Correlations")
                                        
                                        corr_data = []
                                        for corr in correlations:
                                            if isinstance(corr, dict) and not corr.get('error'):
                                                corr_data.append({
                                                    'Feature 1': corr.get('feature1', 'N/A'),
                                                    'Feature 2': corr.get('feature2', 'N/A'),
                                                    'Correlation': f"{corr.get('correlation', 0):.4f}",
                                                    'Strength': corr.get('correlation_strength', 'N/A'),
                                                    'Type': corr.get('correlation_type', 'N/A')
                                                })
                                        
                                        if corr_data:
                                            corr_df = pd.DataFrame(corr_data)
                                            st.dataframe(corr_df, use_container_width=True)
                                        else:
                                            st.info("No strong correlations found between features.")
                                    
                                    # Feature importance (if target available)
                                    feature_importance = feature_patterns.get('feature_importance', {})
                                    if feature_importance and not feature_importance.get('error'):
                                        st.subheader("Feature Importance Analysis")
                                        
                                        top_features = feature_importance.get('top_features', [])
                                        if top_features:
                                            importance_df = pd.DataFrame(top_features)
                                            st.dataframe(importance_df, use_container_width=True)
                            
                            # Relationships Tab
                            with tabs[3]:
                                st.subheader("Feature Relationships")
                                
                                relationships = insights.get('relationships', {})
                                
                                if relationships.get('error'):
                                    st.error(f"{relationships['error']}")
                                else:
                                    # Strong correlations
                                    strong_corr = relationships.get('strong_correlations', [])
                                    if strong_corr:
                                        st.subheader("Strong Correlations")
                                        
                                        corr_data = []
                                        for corr in strong_corr:
                                            if isinstance(corr, dict) and not corr.get('error'):
                                                corr_data.append({
                                                    'Feature 1': corr.get('feature1', 'N/A'),
                                                    'Feature 2': corr.get('feature2', 'N/A'),
                                                    'Pearson Correlation': f"{corr.get('pearson_correlation', 0):.4f}",
                                                    'Spearman Correlation': f"{corr.get('spearman_correlation', 0):.4f}",
                                                    'P-Value': f"{corr.get('p_value', 0):.6f}",
                                                    'Significant': 'Yes' if corr.get('is_significant') else 'No',
                                                    'Strength': corr.get('strength', 'N/A')
                                                })
                                        
                                        if corr_data:
                                            corr_df = pd.DataFrame(corr_data)
                                            st.dataframe(corr_df, use_container_width=True)
                                    
                                    # Multicollinearity warnings
                                    multicollinearity = relationships.get('multicollinearity_warnings', [])
                                    if multicollinearity:
                                        st.subheader("Multicollinearity Warnings")
                                        
                                        for warning in multicollinearity:
                                            if isinstance(warning, dict) and not warning.get('error'):
                                                severity = warning.get('severity', 'low')
                                                features = warning.get('features', [])
                                                
                                                if severity == 'high':
                                                    st.error(f"High multicollinearity: {', '.join(features)}")
                                                else:
                                                    st.warning(f"Moderate multicollinearity: {', '.join(features)}")
                                                
                                                st.write(f"**Recommendation:** {warning.get('recommendation', 'N/A')}")
                                    
                                    # Categorical dependencies
                                    dependencies = relationships.get('dependency_patterns', [])
                                    if dependencies:
                                        st.subheader("Categorical Variable Dependencies")
                                        
                                        dep_data = []
                                        for dep in dependencies:
                                            if isinstance(dep, dict) and not dep.get('error'):
                                                dep_data.append({
                                                    'Feature 1': dep.get('feature1', 'N/A'),
                                                    'Feature 2': dep.get('feature2', 'N/A'),
                                                    "Cramér's V": f"{dep.get('cramers_v', 0):.4f}",
                                                    'P-Value': f"{dep.get('p_value', 0):.6f}",
                                                    'Association Strength': dep.get('association_strength', 'N/A')
                                                })
                                        
                                        if dep_data:
                                            dep_df = pd.DataFrame(dep_data)
                                            st.dataframe(dep_df, use_container_width=True)
                                    
                                    # Mixed-type interactions
                                    interactions = relationships.get('feature_interactions', [])
                                    if interactions:
                                        st.subheader("Cross-Type Feature Interactions")
                                        
                                        int_data = []
                                        for interaction in interactions:
                                            if isinstance(interaction, dict) and not interaction.get('error'):
                                                int_data.append({
                                                    'Categorical Feature': interaction.get('categorical_feature', 'N/A'),
                                                    'Numeric Feature': interaction.get('numeric_feature', 'N/A'),
                                                    'Test Type': interaction.get('test_type', 'N/A'),
                                                    'P-Value': f"{interaction.get('p_value', 0):.6f}",
                                                    'Effect Size': f"{interaction.get('effect_size', 0):.4f}",
                                                    'Effect': interaction.get('effect_interpretation', 'N/A'),
                                                    'Significant': 'Yes' if interaction.get('is_significant') else 'No'
                                                })
                                        
                                        if int_data:
                                            int_df = pd.DataFrame(int_data)
                                            st.dataframe(int_df, use_container_width=True)
                            
                            # Data Quality Tab
                            with tabs[4]:
                                st.subheader("Data Quality Analysis")
                                
                                # Statistical warnings
                                warnings = insights.get('statistical_warnings', [])
                                if warnings:
                                    # Group warnings by severity
                                    high_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'high']
                                    medium_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'medium']
                                    low_warnings = [w for w in warnings if isinstance(w, dict) and w.get('severity') == 'low']
                                    
                                    # Summary metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Critical Issues", len(high_warnings))
                                    with col2:
                                        st.metric("Medium Issues", len(medium_warnings))
                                    with col3:
                                        st.metric("Low Issues", len(low_warnings))
                                    with col4:
                                        total_issues = len(high_warnings) + len(medium_warnings) + len(low_warnings)
                                        st.metric("Total Issues", total_issues)
                                    
                                    # Display warnings by severity
                                    if high_warnings:
                                        st.subheader("Critical Issues (Immediate Action Required)")
                                        for warning in high_warnings:
                                            st.error(f"**{warning.get('type', 'Unknown')}** - {warning.get('message', 'N/A')}")
                                            if 'column' in warning:
                                                st.write(f"**Column:** {warning['column']}")
                                            if 'value' in warning:
                                                st.write(f"**Value:** {warning['value']}")
                                    
                                    if medium_warnings:
                                        st.subheader("Medium Priority Issues")
                                        if show_detailed_warnings:
                                            for warning in medium_warnings:
                                                st.warning(f"**{warning.get('type', 'Unknown')}** - {warning.get('message', 'N/A')}")
                                                if 'column' in warning:
                                                    st.write(f"**Column:** {warning['column']}")
                                        else:
                                            st.warning(f"Found {len(medium_warnings)} medium priority issues. Enable 'Show Detailed Warnings' to see details.")
                                    
                                    if low_warnings and show_detailed_warnings:
                                        st.subheader("Low Priority Issues")
                                        for warning in low_warnings:
                                            st.info(f"**{warning.get('type', 'Unknown')}** - {warning.get('message', 'N/A')}")
                                    elif low_warnings:
                                        st.info(f"Found {len(low_warnings)} low priority issues. Enable 'Show Detailed Warnings' to see details.")
                                
                                # Data quality summary
                                data_quality_issues = insights.get('data_quality', [])
                                if data_quality_issues:
                                    st.subheader("Data Quality Issues")
                                    
                                    for issue in data_quality_issues:
                                        if isinstance(issue, str):
                                            if 'error' in issue.lower():
                                                st.error(f"{issue}")
                                            elif any(keyword in issue.lower() for keyword in ['warning', 'unrealistic', 'inconsistent']):
                                                st.warning(f"{issue}")
                                            else:
                                                st.info(f"{issue}")
                                else:
                                    st.success("No specific data quality issues identified!")
                            
                            # AI Recommendations Tab
                            with tabs[5]:
                                if include_ai_recommendations:
                                    st.subheader("AI-Powered Recommendations")
                                    
                                    ai_recommendations = insights.get('ai_recommendations', {})
                                    
                                    if ai_recommendations.get('error'):
                                        st.error(f"{ai_recommendations['error']}")
                                    else:
                                        # Data preprocessing recommendations
                                        preprocessing = ai_recommendations.get('data_preprocessing', [])
                                        if preprocessing:
                                            st.subheader("Data Preprocessing")
                                            for rec in preprocessing:
                                                if isinstance(rec, dict):
                                                    priority = rec.get('priority', 'low')
                                                    
                                                    if priority == 'high':
                                                        st.error(f"**{rec.get('action', 'N/A')}**")
                                                    elif priority == 'medium':
                                                        st.warning(f"**{rec.get('action', 'N/A')}**")
                                                    else:
                                                        st.info(f"**{rec.get('action', 'N/A')}**")
                                                    
                                                    st.write(f"**Details:** {rec.get('details', 'N/A')}")
                                                    
                                                    if 'specific_steps' in rec:
                                                        with st.expander("View specific steps"):
                                                            for step in rec['specific_steps']:
                                                                st.write(f"• {step}")
                                        
                                        # Feature engineering recommendations
                                        feature_eng = ai_recommendations.get('feature_engineering', [])
                                        if feature_eng:
                                            st.subheader("Feature Engineering")
                                            for rec in feature_eng:
                                                if isinstance(rec, dict):
                                                    priority = rec.get('priority', 'low')
                                                    
                                                    if priority == 'high':
                                                        st.error(f"**{rec.get('action', 'N/A')}**")
                                                    elif priority == 'medium':
                                                        st.warning(f"**{rec.get('action', 'N/A')}**")
                                                    else:
                                                        st.info(f"**{rec.get('action', 'N/A')}**")
                                                    
                                                    st.write(f"**Details:** {rec.get('details', 'N/A')}")
                                                    
                                                    if 'specific_steps' in rec:
                                                        with st.expander("View specific steps"):
                                                            for step in rec['specific_steps']:
                                                                st.write(f"• {step}")
                                        
                                        # Modeling suggestions
                                        modeling = ai_recommendations.get('modeling_suggestions', [])
                                        if modeling:
                                            st.subheader("Modeling Suggestions")
                                            for rec in modeling:
                                                if isinstance(rec, dict):
                                                    priority = rec.get('priority', 'low')
                                                    
                                                    if priority == 'high':
                                                        st.error(f"**{rec.get('action', 'N/A')}**")
                                                    elif priority == 'medium':
                                                        st.warning(f"**{rec.get('action', 'N/A')}**")
                                                    else:
                                                        st.info(f"**{rec.get('action', 'N/A')}**")
                                                    
                                                    st.write(f"**Details:** {rec.get('details', 'N/A')}")
                                                    
                                                    if 'specific_steps' in rec:
                                                        with st.expander("View specific steps"):
                                                            for step in rec['specific_steps']:
                                                                st.write(f"• {step}")
                                        
                                        # Data collection recommendations
                                        data_collection = ai_recommendations.get('data_collection', [])
                                        if data_collection:
                                            st.subheader("Data Collection")
                                            for rec in data_collection:
                                                if isinstance(rec, dict):
                                                    priority = rec.get('priority', 'low')
                                                    
                                                    if priority == 'high':
                                                        st.error(f"**{rec.get('action', 'N/A')}**")
                                                    elif priority == 'medium':
                                                        st.warning(f"**{rec.get('action', 'N/A')}**")
                                                    else:
                                                        st.info(f"**{rec.get('action', 'N/A')}**")
                                                    
                                                    st.write(f"**Details:** {rec.get('details', 'N/A')}")
                                                    
                                                    if 'specific_steps' in rec:
                                                        with st.expander("View specific steps"):
                                                            for step in rec['specific_steps']:
                                                                st.write(f"• {step}")
                                        
                                        # General insights
                                        general = ai_recommendations.get('general_insights', [])
                                        if general:
                                            st.subheader("General Insights")
                                            for insight in general:
                                                if isinstance(insight, dict):
                                                    st.info(f"**{insight.get('category', 'General').title()}:** {insight.get('insight', 'N/A')}")
                                                    st.write(f"**Interpretation:** {insight.get('interpretation', 'N/A')}")
                                                    
                                                    implications = insight.get('implications', [])
                                                    if implications:
                                                        with st.expander("View implications"):
                                                            for implication in implications:
                                                                st.write(f"• {implication}")
                                else:
                                    st.info("AI Recommendations are disabled. Enable them in the configuration section above.")
                            
                            # Business Insights Tab
                            with tabs[6]:
                                if include_business_insights:
                                    st.subheader("💼 Business Impact Analysis")
                                    
                                    business_insights = insights.get('business_insights', [])
                                    
                                    if isinstance(business_insights, list) and business_insights:
                                        for insight in business_insights:
                                            if isinstance(insight, dict):
                                                category = insight.get('category', 'general')
                                                value_score = insight.get('value_score', 'medium')
                                                risk_level = insight.get('risk_level', 'low')
                                                
                                                # Color coding based on value score or risk level
                                                if value_score == 'high' or risk_level == 'low':
                                                    st.success(f"**{category.replace('_', ' ').title()}:** {insight.get('insight', 'N/A')}")
                                                elif value_score == 'low' or risk_level == 'high':
                                                    st.error(f"**{category.replace('_', ' ').title()}:** {insight.get('insight', 'N/A')}")
                                                else:
                                                    st.warning(f"**{category.replace('_', ' ').title()}:** {insight.get('insight', 'N/A')}")
                                                
                                                st.write(f"**Business Impact:** {insight.get('business_impact', 'N/A')}")
                                                
                                                action_items = insight.get('action_items', [])
                                                if action_items:
                                                    with st.expander("Action Items"):
                                                        for action in action_items:
                                                            st.write(f"• {action}")
                                                
                                                # Show additional metrics if available
                                                if 'priority_score' in insight:
                                                    st.write(f"**Priority Score:** {insight['priority_score']:.2f}/1.0")
                                                if 'complexity_score' in insight:
                                                    st.write(f"**Complexity Score:** {insight['complexity_score']:.2f}/1.0")
                                                
                                                st.divider()
                                    
                                    elif isinstance(business_insights, list) and len(business_insights) == 1:
                                        # Handle single error message
                                        error_msg = business_insights[0]
                                        if isinstance(error_msg, str):
                                            st.error(f"{error_msg}")
                                    else:
                                        st.info("No business insights generated.")
                                else:
                                    st.info("Business insights are disabled. Enable them in the configuration section above.")
                            
                            # Export section
                            st.subheader("Export Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Export summary report
                                if st.button("Export Summary Report", use_container_width=True):
                                    summary_report = st.session_state.get('summary_report', '')
                                    if summary_report:
                                        st.download_button(
                                            label="Download Summary Report",
                                            data=summary_report,
                                            file_name=f"advanced_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                            mime="text/plain",
                                            use_container_width=True
                                        )
                            
                            with col2:
                                # Export detailed insights as JSON
                                if st.button("Export Detailed Insights", use_container_width=True):
                                    export_dict = insight_generator.export_insights_to_dict()
                                    
                                    # Convert to JSON string
                                    import json
                                    json_str = json.dumps(export_dict, indent=2, default=str)
                                    
                                    st.download_button(
                                        label="Download Insights JSON",
                                        data=json_str,
                                        file_name=f"advanced_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            
                            with col3:
                                # Export key insights as CSV
                                if st.button("Export Key Insights", use_container_width=True):
                                    key_insights = insights.get('key_insights', [])
                                    if key_insights:
                                        # Convert key insights to DataFrame
                                        insights_data = []
                                        for insight in key_insights:
                                            if isinstance(insight, dict):
                                                insights_data.append({
                                                    'Priority': insight.get('priority', 0),
                                                    'Category': insight.get('category', 'N/A'),
                                                    'Insight': insight.get('insight', 'N/A'),
                                                    'Action Needed': 'Yes' if insight.get('action_needed') else 'No'
                                                })
                                        
                                        if insights_data:
                                            insights_df = pd.DataFrame(insights_data)
                                            csv_str = insights_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="Download Key Insights CSV",
                                                data=csv_str,
                                                file_name=f"key_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                        
                        # Help and documentation
                        with st.expander("Help & Documentation"):
                            st.markdown("""
                            ### AI-Powered Advanced Data Analysis
                            
                            This advanced analysis tool provides comprehensive insights about your dataset using sophisticated statistical methods and AI-powered recommendations.
                            
                            #### **Features:**
                            
                            **Data Composition Analysis:**
                            - Automatic feature type detection (numerical, categorical, datetime, boolean)
                            - Missing data analysis with patterns detection
                            - Memory usage optimization suggestions
                            - Data density and quality metrics
                            
                            **Target Variable Analysis:**
                            - Automatic problem type detection (classification/regression)
                            - Class balance analysis for classification problems
                            - Distribution analysis for regression problems
                            - Statistical significance testing
                            
                            **Feature Pattern Analysis:**
                            - Outlier detection using multiple methods
                            - Skewness and kurtosis analysis
                            - High cardinality detection
                            - Feature correlation analysis
                            
                            **Relationship Discovery:**
                            - Pearson and Spearman correlations
                            - Categorical variable dependencies (Chi-square, Cramér's V)
                            - Mixed-type feature interactions (ANOVA, Kruskal-Wallis)
                            - Multicollinearity detection
                            
                            **Data Quality Assessment:**
                            - Statistical warnings with severity levels
                            - Data consistency checks
                            - Encoding and formatting issue detection
                            - Suspicious value identification
                            
                            **AI Recommendations:**
                            - Data preprocessing suggestions
                            - Feature engineering recommendations
                            - Model selection guidance
                            - Data collection improvements
                            
                            **Business Impact Analysis:**
                            - ROI and priority scoring
                            - Risk assessment
                            - Investment recommendations
                            - Operational complexity analysis
                            
                            #### **How to Use:**
                            1. Upload your dataset using the file uploader
                            2. Select a target variable (optional but recommended)
                            3. Configure analysis options
                            4. Click "Run Comprehensive Analysis"
                            5. Explore results in the tabbed interface
                            6. Export insights for further use
                            
                            #### **Tips for Best Results:**
                            - Ensure your data is in a clean CSV format
                            - Include column headers
                            - Specify the target variable for supervised learning insights
                            - Enable all analysis options for comprehensive results
                            - Review high-priority warnings first
                            
                            #### **Understanding Severity Levels:**
                            - **High (Red):** Immediate action required
                            - **Medium (Yellow):** Attention needed
                            - **Low (Blue):** Minor issues or informational
                            - **Good (Green):** No issues detected
                            """)
                    
                    else:
                        # No data uploaded
                        st.info("Please upload a dataset to begin advanced analysis.")
                        
                        # Sample dataset option
                        st.subheader("Try with Sample Data")
                        st.markdown("Want to see the advanced analysis in action? Try it with sample datasets:")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Load Housing Dataset", use_container_width=True):
                                # Create sample housing dataset
                                np.random.seed(42)
                                
                                n_samples = 1000
                                sample_data = {
                                    'house_age': np.random.normal(15, 8, n_samples),
                                    'income': np.random.lognormal(11, 0.5, n_samples),
                                    'house_size': np.random.normal(2000, 500, n_samples),
                                    'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                                    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.5, 0.3, 0.2]),
                                    'has_garage': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
                                    'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_samples, p=[0.6, 0.25, 0.15]),
                                    'house_price': None  # Will be calculated
                                }
                                
                                # Add some realistic relationships
                                sample_data['house_price'] = (
                                    sample_data['house_size'] * 150 +
                                    sample_data['income'] * 2 +
                                    sample_data['bedrooms'] * 10000 +
                                    np.where(sample_data['location'] == 'Urban', 50000, 0) +
                                    np.where(sample_data['has_garage'], 15000, 0) +
                                    np.random.normal(0, 25000, n_samples)
                                )
                                
                                # Add some missing values
                                missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
                                sample_data['income'][missing_indices] = np.nan
                                
                                sample_df = pd.DataFrame(sample_data)
                                st.session_state['sample_df'] = sample_df
                                st.success("Housing dataset loaded! Go to the Basic EDA tab to see the data.")
                        
                        with col2:
                            if st.button("Load Marketing Dataset", use_container_width=True):
                                # Create sample marketing dataset
                                import numpy as np
                                np.random.seed(123)
                                
                                n_samples = 800
                                sample_data = {
                                    'customer_age': np.random.normal(40, 12, n_samples),
                                    'annual_spending': np.random.lognormal(9, 0.8, n_samples),
                                    'campaign_response': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                                    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
                                    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.3, 0.5, 0.2]),
                                    'num_purchases': np.random.poisson(5, n_samples),
                                    'preferred_channel': np.random.choice(['Online', 'Store', 'Phone'], n_samples, p=[0.5, 0.35, 0.15]),
                                    'satisfaction_score': np.random.normal(7.5, 1.5, n_samples),
                                    'days_since_last_purchase': np.random.exponential(30, n_samples)
                                }
                                
                                # Add some realistic relationships
                                response_prob = (
                                    0.1 + 
                                    (sample_data['satisfaction_score'] - 5) * 0.05 +
                                    np.where(sample_data['education'] == 'PhD', 0.1, 0) +
                                    np.where(sample_data['preferred_channel'] == 'Online', 0.05, 0)
                                )
                                sample_data['campaign_response'] = np.random.binomial(1, np.clip(response_prob, 0, 1), n_samples)
                                
                                # Add missing values
                                missing_indices = np.random.choice(n_samples, int(n_samples * 0.03), replace=False)
                                sample_data['satisfaction_score'][missing_indices] = np.nan
                                
                                sample_df = pd.DataFrame(sample_data)
                                st.session_state['sample_df'] = sample_df
                                st.success("Marketing dataset loaded! Go to the Basic EDA tab to see the data.")
                        
                        st.markdown("""
                        ### What Makes This Analysis Advanced?
                        
                        Our AI-powered analysis goes far beyond basic statistics:
                        
                        **Intelligent Pattern Recognition:**
                        - Automatically detects data types and suggests appropriate analyses
                        - Identifies statistical anomalies and potential data quality issues
                        - Discovers hidden relationships between variables
                        
                        **Comprehensive Statistical Testing:**
                        - Multiple correlation methods (Pearson, Spearman, Cramér's V)
                        - Normality testing with appropriate methods based on sample size
                        - ANOVA and Kruskal-Wallis tests for group comparisons
                        - Chi-square tests for categorical associations
                        
                        **Business-Focused Insights:**
                        - ROI and investment priority scoring
                        - Risk assessment and compliance considerations
                        - Operational complexity analysis
                        - Actionable recommendations with specific implementation steps
                        
                        **Advanced Quality Assurance:**
                        - Multi-method outlier detection (IQR, Z-score, Modified Z-score)
                        - Encoding and formatting issue detection
                        - Multicollinearity analysis with VIF approximation
                        - Missing data pattern analysis
                        
                        **AI-Powered Recommendations:**
                        - Context-aware preprocessing suggestions
                        - Feature engineering recommendations
                        - Model selection guidance based on data characteristics
                        - Data collection improvement strategies
                        """)
                with report_tab:
                    st.subheader("Professional Data Quality Report")

                    st.markdown("""
                    > **Note:** This report is not about giving data insights or business interpretations.
                    > It is a **professional dataset quality check**, similar to what a data analyst or data scientist presents during initial data review — highlighting completeness, consistency, and reliability.
                    """)
                        
                    report = eda_engine.generate_data_quality_report(df, dataset_name=uploaded_file.name)

                    # Display the report summary
                    for key, value in report.items():
                        if key != "exported_csv":
                            st.write(f"**{key}**:")
                            st.write(value)

                    # Show export link if CSV was generated
                    if 'exported_csv' in report:
                        st.success(f"Report exported as: `{report['exported_csv']}`")
# If no file is uploaded
else:
    st.info("Please upload a CSV file to begin.")

