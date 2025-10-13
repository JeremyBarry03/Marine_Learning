
function handleFiles(files) {
    const dropZone = document.getElementById('drop_zone');
    dropZone.style.border = '2px dashed #ccc';

    const file = files[0];
    displayImage(file);
}

function displayImage(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        const dropZone = document.getElementById('drop_zone');
        dropZone.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image" style="max-width: 100%; max-height: 200px;">`;
    };

    reader.readAsDataURL(file);
}

//update the visibility of the "Clear" button based on file selection
function clearFileInput() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        fileInput.value = '';
        document.getElementById('drop_zone').innerHTML = '<p>Upload an image.</p>';
        document.getElementById('clearButton').style.display = 'none';
        document.getElementById('fruit-name').innerHTML = "";
        document.getElementById('fruit-nutrition').innerHTML = "";
    }
}

document.getElementById('fileInput').addEventListener('change', function() {
    const clearButton = document.getElementById('clearButton');
    clearButton.style.display = this.files.length > 0 ? 'block' : 'none';
});

//send image data to model for prediction
function predictImage() {
    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors',
    })
    .then(response => response.json())
    .then(data => {
        console.log('Prediction:', data.prediction);
        console.log('Nutrition:', data.nutrition);
        if(data.nutrition.calories == undefined){
            nutrition = "There are no nutrition facts available for this fruit."
        }else{
            nutrition = "<b>Nutrition Facts</b><br>Calories: " + data.nutrition.calories + "<br>Carbs: " + data.nutrition.carbohydrates + "g<br>Fat: " + data.nutrition.fat + "g<br>Protein: " + data.nutrition.protein + "g<br>Sugar: " + data.nutrition.sugar + "g"
        }
        prediction = data.prediction
        document.getElementById('fruit-name').innerHTML = prediction;
        document.getElementById('fruit-nutrition').innerHTML = nutrition;
        
    })
    .catch(error => console.error('Error:', error));
}

//------------------------------------------------------
// DEBUGGING AND CODE FOR DRAGGING AND DROPPING AN IMAGE
//------------------------------------------------------

// function dropHandler(ev) {
//     console.log("File(s) dropped");
//     document.getElementById('drop_zone').classList.remove('over');
//     ev.preventDefault();

//     let files = [];

//     if (ev.dataTransfer.items) {
//         files = [...ev.dataTransfer.items].map(item => item.kind === "file" ? item.getAsFile() : null);
//     } else {
//         files = [...ev.dataTransfer.files];
//     }

//     //filter out non-image files and limit to one file
//     files = files.filter(file => file && file.type.startsWith('image/')).slice(0, 1);

//     //for debug
//     if (files.length === 1) {
//         console.log(`… image file name = ${files[0].name}`);
//         clearButton.style.display =  'block';
//         handleFiles(files);
//     } else {
//         console.log("Please drop only one image file.");
//     }
// }

// function dragOverHandler(ev) {
//     ev.preventDefault();
// }

// function dragEnterHandler(ev) {
//     document.getElementById('drop_zone').classList.add('over');
// }

// function dragLeaveHandler(ev) {
//     document.getElementById('drop_zone').classList.remove('over');
// }

//buttons file processing
//function to handle button click
// function logButtonPress(event) {
//     console.log("Button pressed:", event.target.id);
// }
//document.getElementById("uploadButton").addEventListener("click", browseFiles);
//document.getElementById("submitButton").addEventListener("click", logButtonPress);

