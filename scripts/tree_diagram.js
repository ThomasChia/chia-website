import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

const pythonSkills = [
    "Python/PyTorch",
    "Python/Pandas",
    "Python/Flask",
]

const databaseSkills = [
    "SQL/MySQL",
    "SQL/PostgreSQL",
    // "SQL/MongoDB",
]

const frontendSkills = [
    "Frontend/HTML",
    "Frontend/CSS",
    "Frontend/JavaScript",
]

const pythonPlot =Plot.plot({
    axis: null,
    height: 100,
    width: 350,
    margin: 10,
    marginLeft: 70,
    marginRight: 60,
    marks: [
        Plot.tree(pythonSkills, {textStroke: "white"})
    ],
    color: {
        scheme: "accent" // use the "accent" scheme
      },
    style: {fontFamily: "Arial", fontSize: "14px"},
    })

const databasePlot =Plot.plot({
    axis: null,
    height: 100,
    width: 350,
    margin: 10,
    marginLeft: 90,
    marginRight: 90,
    marks: [
        Plot.tree(databaseSkills, {textStroke: "white"})
    ],
    style: {fontFamily: "Arial", fontSize: "14px"},
    })

const frontendPlot =Plot.plot({
    axis: null,
    height: 100,
    width: 350,
    margin: 10,
    marginLeft: 90,
    marginRight: 90,
    marks: [
        Plot.tree(frontendSkills, {textStroke: "white"})
    ],
    color: {
        scheme: "oranges"
      },
    style: {fontFamily: "Arial", fontSize: "14px"},
    })

// const plot = Plot.rectY({length: 10000}, Plot.binX({y: "count"}, {x: Math.random})).plot();
const pythonPlotDiv = document.getElementById("pythonPlot");
const databasePlotDiv = document.getElementById("databasePlot");
const frontendPlotDiv = document.getElementById("frontendPlot");
pythonPlotDiv.append(pythonPlot);
databasePlotDiv.append(databasePlot);
frontendPlotDiv.append(frontendPlot);
