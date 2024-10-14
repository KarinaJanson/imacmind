import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np

# Function to upload and read the Excel file
def upload_file():
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Dataframe preview:")
        st.write(df.head())
        return df
    return None

# Function to select columns for the plot
def select_columns(df):
    return st.multiselect("Select columns for the plot", df.columns.tolist())

# Function to select plot type
def select_plot_type():
    return st.selectbox("Select Plot Type", ["Sunburst", "Treemap", "Sankey", "Item Similarity Network"])

# Function to validate the column selection
def validate_columns(columns, plot_type):
    if plot_type == "Sankey" and len(columns) < 2:
        st.warning("Please select at least 2 columns for the Sankey plot.")
        return False
    elif plot_type in ["Sunburst", "Treemap"] and len(columns) < 3:
        st.warning("Please select at least 3 columns for Sunburst and Treemap plots.")
        return False
    elif plot_type == "Item Similarity Network" and len(columns) < 1:
        st.warning("Please select the column containing item text for the Item Similarity Network plot.")
        return False
    return True

# Function to select layers for the plot
def select_layers(columns):
    col1 = st.selectbox("Select First Layer", columns, key='col1')
    col2 = st.selectbox("Select Second Layer", columns, key='col2')
    col3 = st.selectbox("Select Third Layer", columns, key='col3')
    col4 = st.selectbox("Select Fourth Layer (optional)", [None] + columns, key='col4')
    return col1, col2, col3, col4

# Function to preprocess the DataFrame for hierarchical plots
def preprocess_dataframe_for_hierarchy(df, columns):
    path_cols = columns
    df_grouped = df.groupby(path_cols).size().reset_index(name='value')
    return df_grouped

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to generate and display the plot with a fixed color palette
def generate_plot(df, plot_type, col1, col2, col3, col4, color_var=None):
    color_discrete_sequence = px.colors.qualitative.Plotly  # Fixed color palette
    columns = [col for col in [col1, col2, col3, col4] if col]
    
    if plot_type in ["Sunburst", "Treemap"]:
        df_preprocessed = preprocess_dataframe_for_hierarchy(df, columns)
        if plot_type == "Sunburst":
            fig = px.sunburst(df_preprocessed, path=columns, values='value', color_discrete_sequence=color_discrete_sequence)
        elif plot_type == "Treemap":
            fig = px.treemap(df_preprocessed, path=columns, values='value', color_discrete_sequence=color_discrete_sequence)
    elif plot_type == "Sankey":
        fig = generate_sankey(df, columns)
    elif plot_type == "Item Similarity Network":
        fig = generate_item_similarity_network(df, col1, color_var)
    
    st.plotly_chart(fig)
    return fig

# Function to generate Sankey plot with dynamic columns
def generate_sankey(df, columns):
    all_labels = pd.concat([df[col] for col in columns]).unique()
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    sources = []
    targets = []
    values = []
    
    for i in range(len(columns) - 1):
        source_col = columns[i]
        target_col = columns[i + 1]
        
        grouped = df.groupby([source_col, target_col]).size().reset_index(name='value')
        
        sources.extend(grouped[source_col].map(label_map))
        targets.extend(grouped[target_col].map(label_map))
        values.extend(grouped['value'])

    link = dict(source=sources, target=targets, value=values)
    node = dict(label=list(label_map.keys()), pad=15, thickness=20)

    fig = go.Figure(data=[go.Sankey(link=link, node=node)])
    fig.update_layout(title_text="Sankey Diagram", font_size=10)
    return fig

# Function to generate item similarity network
def generate_item_similarity_network(df, text_column, color_var):
    df['processed_text'] = df[text_column].apply(preprocess_text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    cosine_similarities = cosine_similarity(tfidf_matrix)

    G = nx.Graph()
    items = df[text_column].tolist()
    
    if color_var:
        unique_groups = df[color_var].unique()
        color_map = {group: color for group, color in zip(unique_groups, px.colors.qualitative.Plotly)}
    else:
        color_map = {item: 'blue' for item in items}

    for i, item in enumerate(items):
        G.add_node(i, label=item, color=color_map[df[color_var].iloc[i]] if color_var else 'blue')

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if cosine_similarities[i][j] > 0.5:  # Example threshold for similarity
                G.add_edge(i, j, weight=cosine_similarities[i][j])

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.get_edge_data(edge[0], edge[1])['weight']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(weight)

    node_trace = go.Scatter(
        x=[pos[k][0] for k in range(len(items))],
        y=[pos[k][1] for k in range(len(items))],
        text=[item for item in items],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True if color_var else False,
            colorscale='Viridis',
            size=10,
            color=[G.nodes[k]['color'] for k in range(len(items))],
            line_width=2
        )
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        mode='lines'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Item Similarity Network',
                        titlefont_size=16,
                        showlegend=True if color_var else False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )]
                    ))

    # Add color scale legend if color_var is specified
    if color_var:
        colors = list(color_map.values())
        fig.update_layout(
            coloraxis=dict(
                colorscale='Viridis',
                colorbar=dict(
                    title=color_var,
                    tickvals=list(color_map.values()),
                    ticktext=list(color_map.keys())
                )
            )
        )

    return fig

# Function to save the plot as an HTML file with a timestamped filename
def save_plot_as_html(fig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{timestamp}.html"
    pio.write_html(fig, file=filename, auto_open=False)
    st.success(f"Plot saved as {filename}")

# Function to save similarity scores as an Excel file
def save_similarity_scores(df, cosine_similarities, items, questionnaire_column, optional_column=None):
    rows = []
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i < j:
                similarity_score = cosine_similarities[i][j]
                questionnaire_item1 = df[questionnaire_column].iloc[i]
                questionnaire_item2 = df[questionnaire_column].iloc[j]

                optional_info_item1 = df[optional_column].iloc[i] if optional_column else None
                optional_info_item2 = df[optional_column].iloc[j] if optional_column else None

                row = [item1, item2, similarity_score, questionnaire_item1, questionnaire_item2]
                if optional_column:
                    row.extend([optional_info_item1, optional_info_item2])

                rows.append(row)

    columns = ["Item 1", "Item 2", "Similarity Score", f"{questionnaire_column} of Item 1", f"{questionnaire_column} of Item 2"]
    if optional_column:
        columns.extend([f"{optional_column} of Item 1", f"{optional_column} of Item 2"])

    df_similarity = pd.DataFrame(rows, columns=columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"similarity_scores_{timestamp}.xlsx"
    df_similarity.to_excel(filename, index=False)
    st.success(f"Similarity scores saved as {filename}")

# Function to reset selectio
def reset_selections():
    st.button("Reset Selections", on_click=reset_columns)

# Function to reset column selections (stub for illustration)
def reset_columns():
    st.experimental_rerun()

# Function to style the application
def style_app():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Title of the Streamlit app
st.title("ItemComplex: framework for ex-post harmonization across  multi-item instrument data")

# Apply custom styles
style_app()

# Display user instructions
st.sidebar.markdown(
    """
    ## Instructions
    1. **Upload** your Excel file containing the questionnaire data.
    2. **Select** the type of plot you want to generate.
    3. **Choose** columns relevant to the selected plot type.
    4. For **Item Similarity Network**, select the text column and optionally a color variable.
    5. Click **Generate Plot** to view your plot.
    6. Use the **Save Plot as HTML** button to save the generated plot.
    7. Use the **Generate Similarity Network** button to compute and save similarity scores.
    8. Click **Reset Selections** to clear the current selections.
    """
)

# Upload file and display dataframe
df = upload_file()

if df is not None:
    # Select plot type early to ensure it's defined
    plot_type = select_plot_type()

    if plot_type == "Item Similarity Network":
        st.write("### Select Columns for Item Similarity Network")
        item_column = st.selectbox("Select the column containing item text", df.columns)
        questionnaire_column = st.selectbox("Select the column containing questionnaire names", df.columns)
        optional_column = st.selectbox("Select an optional column (e.g., construct)", [None] + df.columns.tolist())

        if st.button("Generate Similarity Network"):
            fig = generate_item_similarity_network(df, item_column, optional_column)
            st.plotly_chart(fig)
            st.session_state['similarity_fig'] = fig

            # Compute similarity scores
            cosine_similarities = cosine_similarity(TfidfVectorizer().fit_transform(df[item_column].apply(preprocess_text)))
            
            # Save similarity scores with the new structure
            save_similarity_scores(df, cosine_similarities, df[item_column].tolist(), questionnaire_column, optional_column)

    else:
        # Select columns for the plot
        columns = select_columns(df)

        if columns:
            if validate_columns(columns, plot_type):
                if plot_type in ["Sunburst", "Treemap", "Sankey"]:
                    st.write("### Select Layers for the Plot")
                    col1, col2, col3, col4 = select_layers(columns)
                    color_var = None  # Explicitly set color_var to None for these plot types
                else:
                    col1 = columns[0]
                    col2 = col3 = col4 = None

                    # Select color variable only for Item Similarity Network
                    color_var = st.selectbox("Select Color Variable (optional)", [None] + columns, key='color_var')

                # Button to generate the plot
                if st.button(f"Generate {plot_type} Plot"):
                    fig = generate_plot(df, plot_type, col1, col2, col3, col4, color_var)
                    # Save the figure to session state
                    st.session_state['fig'] = fig

                # Display save plot button only if a plot has been generated
                if 'fig' in st.session_state:
                    if st.button("Save Plot as HTML"):
                        save_plot_as_html(st.session_state['fig'])

    # Reset selections
    reset_selections()
