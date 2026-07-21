import plotly.io as pio
import plotly.graph_objects as go


# Option A: Force browser rendering (opens a new tab)
pio.renderers.default = "browser"

# Option B: Use the built-in vscode renderer if you are in VS Code
pio.renderers.default = "vscode"
fig = go.Figure()

fig.show()