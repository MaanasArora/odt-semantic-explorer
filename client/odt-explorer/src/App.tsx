import { useState, useEffect } from "react";
import "./App.css";

interface Dataset {
  id: number;
  name: string;
}

interface Column {
  id: number;
  name: string;
  dataset: Dataset;
}

interface Domain {
  id: number;
  columns: Column[];
}

function App() {
  const [domains, setDomains] = useState<Domain[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/domains")
      .then((response) => response.json())
      .then((data) => setDomains(data));
  }, []);

  return (
    <div className="App">
      <h1>ODT Explorer</h1>
      {domains.map((domain) => (
        <div style={{ display: "flex", flexDirection: "row" }}>
          <h2 style={{ flex: 1 }}>Domain {domain.id}</h2>
          <ul style={{ flex: 2, listStyleType: "none" }}>
            {domain.columns.map((column) => (
              <li key={column.id}>{column.dataset.name}</li>
            ))}
          </ul>
          <ul style={{ flex: 2, listStyleType: "none" }}>
            {domain.columns.map((column) => (
              <li key={column.id}>{column.name}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default App;
