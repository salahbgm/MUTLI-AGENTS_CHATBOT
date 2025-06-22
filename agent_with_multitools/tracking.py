# tracking.py

class ToolTracker:
    def __init__(self):
        self.tools_used = []  # liste de noms
        self.tool_results = {}  # mapping nom -> résultat (optionnel)

    def add_tool(self, tool_name, result=None):
        if tool_name and tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if result:
            self.tool_results[tool_name] = result

    def get_tools(self):
        return self.tools_used

    def get_tools_string(self, contributing_only=False, answer=None):
        if not self.tools_used:
            return "Aucun"
        if contributing_only and answer:
            filtered = []
            for tool in self.tools_used:
                result = self.tool_results.get(tool)
                # Si c'est une string, on compare directement
                if isinstance(result, str) and result in answer:
                    filtered.append(tool)
                # Si c'est un dico, on cherche une string dans ses valeurs
                elif isinstance(result, dict):
                    for v in result.values():
                        if isinstance(v, str) and v in answer:
                            filtered.append(tool)
                            break
            return ", ".join(filtered) if filtered else "Aucun"
        return ", ".join(self.tools_used)


    def reset(self):
        self.tools_used = []
        self.tool_results = {}

# Instance singleton
tracker = ToolTracker()