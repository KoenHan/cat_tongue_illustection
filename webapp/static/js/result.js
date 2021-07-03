$(() => {
    setInterval(listenImage, 10000);
})

async function listenImage() {
    const defaultUrl = "/static/image/loading_cat.gif"
    const img_results = document.getElementsByClassName("img_results");
    for (let i = 0; i < img_results.length; ++i) {
        let img = img_results[i];
        const parser = new URL(img.src);
        console.log(parser.pathname);
        if (parser.pathname !== defaultUrl) {
            console.log("Skip!");
            continue;
        }
        console.log("Requesting edited image.");
        $.ajax({
            url: img.alt,
            type: 'GET',
            dataType: 'text'
        })
        .done(function(_) {
            console.log(`Reached ${img.alt}`);
            img.src = img.alt;
            const download_link = document.getElementById(`download_link_${i}`);
            download_link.style.visibility = 'visible';
        })
        .fail(function(_) {
            console.log("Cannot reach.")
        })
    }
}
