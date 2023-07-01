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

var checkDead = setInterval(function () {
    var characterTop = parseInt(window.getComputedStyle(character).getPropertyValue("top"));
    var blockLeft = parseInt(window.getComputedStyle(block).getPropertyValue("left"));

    if (blockLeft < 20 && blockLeft > 0 && characterTop >= 130) {
        block.style.animation = "none";
        block.style.display = "none";
        alert("You lose!");
        ShowResetButton();
    }
}, 10);

var resetButton = document.getElementById("resetButton");
resetButton.addEventListener("click", function () {
    block.style.animation = "block 1s infinite linear";
    block.style.display = "block";
    resetButton.style.display = "none";
});