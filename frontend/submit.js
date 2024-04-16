const form = document.getElementById("image-form");
form.addEventListener("submit", formSubmit);

async function formSubmit(e){
    e.preventDefault()
    try{
        var input = document.getElementById('imageInput')
        var data = new FormData()
        data.append("file", input.files[0])

        const response = await fetch('http://127.0.0.1:5000/post-image', {
            method: 'POST',
            body: data
        })
                    
        const responseData = await response.json();
        console.log(responseData)
        document.getElementById("result").innerHTML = "Результат: " + responseData.result;
    }
    catch (error){
        console.log(error.message)
    }
}