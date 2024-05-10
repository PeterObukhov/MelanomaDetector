const form = document.getElementById("image-form");
form.addEventListener("submit", formSubmit);

good = ["Меланоцитарный невус", "Доброкачественное кератоподобное образование"]

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
        var result = responseData.result;
        document.getElementById("result").innerHTML = "РЕЗУЛЬТАТ: " + result.toUpperCase();
        if(good.includes(result)){
            document.getElementById("result").style.color = 'green';
        }
        else{
            document.getElementById("result").style.color = 'red';

        }
        document.getElementById("result").style.display = 'block'; 
    }
    catch (error){
        console.log(error.message)
    }
}