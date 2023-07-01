var character = document.getElementById("character");
var block = document.getElementById("block");

function jump() {
    if (character.classList != "animate") {
        character.classList.add("animate");
    }
    setTimeout(function () {
        character.classList.remove("animate");
    }, 500);
}

function ShowResetButton() {
    document.getElementById("resetButton").style.display = "block";
}

function handleKeyDown(event) {
    if (event.code === "Space") {
        jump();
    }
}

function getRandomLeft() {
    var maxWidth = window.innerWidth - block.offsetWidth;
    return Math.floor(Math.random() * maxWidth);
}

var blocks = [];
var maxBlocks = 3;
var score = 0;
var scoreElement = document.getElementById("score");
var scoreIncrement = false;
var checkDeadInterval;

function createBlock() {
    var newBlock = document.createElement("div");
    newBlock.classList.add("block");
    newBlock.style.left = getRandomLeft() + "px";
    document.body.appendChild(newBlock);
    blocks.push(newBlock);
}
  
function removeBlock(block) {
    block.remove();
    var index = blocks.indexOf(block);
    if (index > -1) {
      blocks.splice(index, 1);
    }
}

function checkDead() {
    checkDeadInterval = setInterval(function () {
        var characterTop = parseInt(window.getComputedStyle(character).getPropertyValue("top"));
        var blockLeft = parseInt(window.getComputedStyle(block).getPropertyValue("left"));
        var blockWidth = parseInt(window.getComputedStyle(block).getPropertyValue("width"));

        if (blockLeft < blockWidth && characterTop >= 130) {
            clearInterval(checkDeadInterval);
            block.style.animation = "none";
            block.style.display = "none";
            alert("You lose!");
            ShowResetButton();
        } else if (blockLeft <= 20 && !scoreIncrement) {
            score++;
            scoreElement.textContent = "Score: " + score;
            scoreIncrement = true;
        } else if (blockLeft > 0) {
            scoreIncrement = false;
        }
    }, 10);
}

var resetButton = document.getElementById("resetButton");
resetButton.addEventListener("click", function () {
    block.style.animation = "block 1s infinite linear";
    block.style.display = "block";
    resetButton.style.display = "none";
    score = 0;
    scoreElement.textContent = "Score: " + score;
    scoreIncrement = false;
    clearInterval(checkDeadInterval);
    checkDead()
});

checkDead();