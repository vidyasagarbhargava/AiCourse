
let btn = document.getElementById('btn')
console.log(btn)
let count = 0
btn.innerText = 'I havent been clicked'
btn.onclick = () => {
    console.log('i was clicked')
    count += 1
    btn.innerText = `I have been clicked ${count} times`
}
btn.onmouseenter = () => console.log('mouse entered')

// let x = 4
// let y = 10

// console.log(x)

// y = x + 3
// console.log(y)

// console.log('hello world')

// let myfunction = (x) => {
//     x += 1
//     console.log(x)
//     return x
// }

// function myfunc(x) {
//     console.log(x)
// }

// z = myfunction(y)
// console.log('z:', z)

// for (let i = 0; i < 10; i++) {
//     console.log(i)
// }

// let i = 0
// while (i < 10) {
//     console.log(i)
//     i += 1
// }

// class MyClass {
    
//     constructor() {}

//     func = () => {console.log('yo')}
// }

// let c = new MyClass()
// c.func()