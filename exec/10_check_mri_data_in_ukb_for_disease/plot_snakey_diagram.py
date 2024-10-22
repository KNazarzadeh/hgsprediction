import pandas as pd
import plotly.graph_objects as go

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
##################################################
# Sample data as a list of dictionaries
data = [
    {"source": "All Stroke Patients", "target": "No MRI", "value": 31},
    {"source": "All Stroke Patients", "target": "Available MRI", "value": 25},
    {"source": "Available MRI", "target": "non Motor Lesions", "value": 12},
    {"source": "Available MRI", "target": "Motor Lesions", "value": 14},
    {"source": "Motor Lesions", "target": "Bilateral", "value": 1},
    {"source": "Motor Lesions", "target": "Right hemespheir (RH)", "value": 4},
    {"source": "Motor Lesions", "target": "Left hemespheir (LF)", "value": 9},
    {"source": "Bilateral", "target": "Bi - Right Dominat hand", "value": 1},
    {"source": "Bilateral", "target": "Bi - Left Dominat hand", "value": 0},
    {"source": "Right hemespheir (RH)", "target": "RH - Right Dominat hand", "value": 4},
    {"source": "Right hemespheir (RH)", "target": "RH - Left Dominat hand", "value": 0},
    {"source": "Left hemespheir (LF)", "target": "LH - Right Dominat hand", "value": 8},
    {"source": "Left hemespheir (LF)", "target": "LH - Left Dominat hand", "value": 1},
    {"source": "LH - Right Dominat hand", "target": "decreased HGS after disease", "value": 5},
    {"source": "LH - Right Dominat hand", "target": "increased HGS after disease", "value": 3},
]
##################################################
# Appending data to the DataFrame
df = pd.DataFrame(data)

# Displaying the DataFrame
print(df)

##################################################
# Create lists for the Sankey plot
sources = df['source'].tolist()
targets = df['target'].tolist()
values = df['value'].tolist()
##################################################
# Create a list of unique nodes
all_nodes = list(pd.unique(sources + targets))
##################################################
# Map each source and target to its index in the node list
# source_indices = [all_nodes.index(src) for src in sources]
source_indices = [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 9]
target_indices = [all_nodes.index(tgt) for tgt in targets]
##################################################
# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=200,
        thickness=20,
        line=dict(color="black", width=.5),
        label=all_nodes,
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values,
        hovertemplate='Value: %{value}'  # Add hover value labels
    )
))

# Save the Sankey plot as a PNG image
fig.write_image("sankey_plot.png")

print("===== Done! =====")
embed(globals(), locals())