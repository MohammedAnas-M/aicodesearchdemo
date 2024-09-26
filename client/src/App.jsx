import { useEffect, useState } from "react";

function App() {
  const [data, setData] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const getPrompt = (e) => {
    setPrompt(e.target.value);
  };

  const sendPrompt = () => {
    setData([]);
    setIsLoading(true);
    fetch("/members", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prompt),
    })
      .then((res) => {
        return res.json();
      })
      .then((data) => {
        console.log("Data ", data);
        setData(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.log("Error while fetching data from server ", err);
      });
  };

  return (
    <>
      <h1>AI Code Search</h1>
      <div>
        <h4>{prompt}</h4>
        <br />
        <label>Prompt : </label>{" "}
        <textarea
          style={{ width: "100%", height: "300px" }}
          onChange={(e) => getPrompt(e)}
          value={prompt}
          type="text"
        />
        <button onClick={sendPrompt}>Send</button>
        <br />
        <h4>Response :</h4>
        <br />
        {isLoading ? "Loading ..." : ""}
        <div style={{maxWidth:'100rem',overflowY:"scroll"}}>
          {" "}
          <pre>{data?.members}</pre>
        </div>
      </div>
    </>
  );
}

export default App;
