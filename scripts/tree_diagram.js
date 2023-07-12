import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

const pythonSkills = [
    "Python/PyTorch",
    "Python/Pandas",
    "Python/Flask",
]

const databaseSkills = [
    "SQL/MySQL",
    "SQL/PostgreSQL",
    "SQL/MongoDB",
]

const frontendSkills = [
    "Frontend/HTML",
    "Frontend/CSS",
    "Frontend/JavaScript",
]

const pythonPlot =Plot.plot({
    axis: null,
    height: 100,
    margin: 10,
    marginLeft: 40,
    marginRight: 120,
    marks: [
        Plot.tree(pythonSkills, {textStroke: "white"})
    ]
    })

const databasePlot =Plot.plot({
    axis: null,
    height: 100,
    margin: 10,
    marginLeft: 40,
    marginRight: 120,
    marks: [
        Plot.tree(databaseSkills, {textStroke: "white"})
    ]
    })

// const plot = Plot.rectY({length: 10000}, Plot.binX({y: "count"}, {x: Math.random})).plot();
const div = document.getElementById("myplot");
div.append(pythonPlot);
div.append(databasePlot);
console.log("Hello from tree_diagram.js")
