const fileElem = document.getElementById("file");

fileElem.addEventListener('change', (event) => {
    const files = event.target.files;
    const filenames = [];
    for (let i = 0; i < files.length; ++i) {
        filenames.push(files.item(i).name);
    }
    const new_message = filenames.map(name => {
        return name.replace(/&/g, '&amp;')
                   .replace(/</g, '&lt;')
                   .replace(/>/g, '&gt;')
                   .replace(/"/g, '&quot;')
                   .replace(/'/g, '&#39;');
    }).join("\n");
    console.log(new_message);

    document.getElementById("message").innerText = new_message;
})
